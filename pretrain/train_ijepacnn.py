# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import timm

from lightly.data import LightlyDataset
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.ijepa_transform import IJEPATransform
# from lightly.models.utils import get_weight_decay_parameters
# from lightly.utils.lars import LARS
# from lightly.utils.scheduler import CosineWarmupScheduler

from timm.models.layers import trunc_normal_
from timm.layers import LayerNorm2d

from pretrain.trainer_common import LightlyModelMomentum, main_pretrain

import models.sparse_encoder as sparse_encoder

from pretrain.metrics import contrastive_acc_eval, eval_feature_descriptors
from pretrain.online_classification_benchmark import OnlineLinearClassificationBenckmark
from pretrain.ijepa_mask import MultiBlockMask

# import models.X does more than just importing the `models` module!
# It also replaces some convnext models in the `timm` model registry with a ConvNext implementation that supports sparsity.
import models.convnext

class IJEPA_CNN(LightlyModelMomentum):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if hasattr(self.backbone, 'sparse'):
            self.backbone.sparse = True
            self.backbone_momentum.sparse = False
        self.backbone_sparse = sparse_encoder.dense_model_to_sparse(self.backbone)

        # The sparse backbone doesn't work for online eval (TODO_ fix this)
        # For now use the momentum backbone for online eval
        self.backbone_for_online_eval = self.backbone_momentum

        self.mask_token = nn.Parameter(torch.zeros(1, self.backbone.num_features, 1, 1))
        trunc_normal_(self.mask_token, mean=0, std=.02, a=-.02, b=.02)

        if self.cfg.backbone.name.lower().startswith('resnet') or self.cfg.backbone.name.lower().startswith('wide_resnet'):
            norm_cls = nn.BatchNorm2d
        elif self.cfg.backbone.name.lower().startswith('convnext'):
            norm_cls = LayerNorm2d

        if self.cfg.use_projection_head:
            proj_layers = []
            for i in range(1):
                proj_layers.append(nn.Conv2d(self.backbone.num_features, 
                                            self.backbone.num_features, 
                                            kernel_size=1,
                                            padding='same'))
                proj_layers.append(norm_cls(self.backbone.num_features))
                proj_layers.append(nn.ReLU(inplace=True))
            self.projection_head = nn.Sequential(*proj_layers)

            self.projection_head_momentum = copy.deepcopy(self.projection_head)
            deactivate_requires_grad(self.projection_head_momentum)

            self.projection_head = sparse_encoder.dense_model_to_sparse(self.projection_head)
           
        pred_layers = []
        for i in range(self.cfg.predictor.n_layers):
            if self.cfg.predictor.get("dw_sep_conv", False):
                # Depthwise separable convolution
                pred_layers.append(nn.Conv2d(self.backbone.num_features, 
                                             self.backbone.num_features, 
                                             kernel_size=self.cfg.predictor.kernel_size,
                                             padding='same',
                                             groups=self.backbone.num_features))
                pred_layers.append(nn.Conv2d(self.backbone.num_features, 
                                             self.backbone.num_features, 
                                             kernel_size=1,
                                             padding='same'))
            else:
                pred_layers.append(nn.Conv2d(self.backbone.num_features, 
                                            self.backbone.num_features, 
                                            kernel_size=self.cfg.predictor.kernel_size,
                                            padding='same'))
            pred_layers.append(norm_cls(self.backbone.num_features))
            pred_layers.append(nn.ReLU(inplace=True))
        self.predictor = nn.Sequential(*pred_layers)

        self.criterion = F.smooth_l1_loss

    def setup_transform(self):
        self.transform = IJEPATransform(self.input_size)

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.cfg.backbone.name.lower().startswith('resnet') or self.cfg.backbone.name.lower().startswith('wide_resnet'):
            self.downsample_raito = 32
        else:
            self.downsample_raito = self.backbone.get_downsample_ratio()
        # if self.cfg.mask.strategy == "random":
        self.fmap_h, self.fmap_w = self.input_size // self.downsample_raito, self.input_size //self. downsample_raito
        self.len_keep = round(self.fmap_h * self.fmap_w * (1 - self.cfg.mask_ratio))
        # elif self.cfg.mask.strategy == "multi-block":
        self.multi_block_mask = MultiBlockMask(
            input_size=self.input_size,
            patch_size=self.downsample_raito,
            **self.cfg.mask.mutli_block_kwargs
        )

    def get_views_to_log_from_batch(self, batch):
        inp_bchw = batch[0]
        context_mask_b1ff, target_mask_b1ff = self.mask(inp_bchw.shape[0], inp_bchw.device)  # (B, 1, f, f)
        context_mask_b1hw = context_mask_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
        target_mask_b1hw  =  target_mask_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
        context_bchw = inp_bchw * context_mask_b1hw
        target_bchw = inp_bchw * target_mask_b1hw
        return [inp_bchw, context_bchw, target_bchw]
    
    def contrastive_acc_eval(self, dataset, file_paths=None):
        sparse_encoder._cur_active = torch.ones_like(sparse_encoder._cur_active)
        return contrastive_acc_eval(self.backbone_momentum, dataset, input_size=self.input_size)
    
    def eval_feature_descriptors(self, dataset):
        sparse_encoder._cur_active = torch.ones_like(sparse_encoder._cur_active)
        return eval_feature_descriptors(
            self.backbone_momentum,
            dataset,
            cfg_name=self.cfg.name,
            current_epoch=self.current_epoch,
        )

    # def on_validation_epoch_end(self) -> None:
    #     mask_shape = sparse_encoder._cur_active.shape
    #     sparse_encoder._cur_active = torch.ones((1, 1, mask_shape[2], mask_shape[3]),
    #                                              device=sparse_encoder._cur_active.device)
    #     super().on_validation_epoch_end()
    
    def mask(self, B: int, device, generator=None):
        if self.cfg.mask.strategy == "mixed":
            if torch.rand(1) < self.cfg.mask.mixed_mutli_block_ratio:
                strategy = "multi-block"
            else:
                strategy = "random"
        else:
            strategy = self.cfg.mask.strategy
        if strategy == "random":
            h, w = self.fmap_h, self.fmap_w
            idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
            idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
            context_mask = torch.zeros(B, h * w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w)
            target_mask = context_mask.logical_not()
            return context_mask, target_mask   
        elif strategy == "multi-block":
            context_mask, target_mask = self.multi_block_mask(B)
            context_mask = context_mask.unsqueeze(1).to(device, dtype=torch.bool)
            target_mask = target_mask.unsqueeze(1).to(device, dtype=torch.bool)
            return context_mask, target_mask
               

    def forward(self, x):
        inp_bchw = x
        # step1. Mask
        context_mask_b1ff, target_mask_b1ff  = self.mask(inp_bchw.shape[0], inp_bchw.device)  # (B, 1, f, f)
        sparse_encoder._cur_active = context_mask_b1ff    # (B, 1, f, f)
        active_b1hw = context_mask_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
        masked_bchw = inp_bchw * active_b1hw
        
        # step2. Encode
        features_bcff = self.backbone_sparse(masked_bchw)

        # step 3. Project
        if self.projection_head is not None:
            features_bcff = self.projection_head(features_bcff)
        
        # step 4. Fill-in mask tokens
        mask_tokens = self.mask_token.expand_as(features_bcff) # expands singleton dimensions to match the shape of features_bcff
        # where context_mask_b1ff is True, use features_bcff, where it's False (i.e. where it masked out a patch) use mask_tokens
        features_m_bcff = torch.where(context_mask_b1ff.expand_as(features_bcff), features_bcff, mask_tokens.to(features_bcff.dtype))   # fill in empty (non-active) positions with [mask] tokens

        z = self.predictor(features_m_bcff)

        return z, context_mask_b1ff, target_mask_b1ff

    def forward_momentum(self, x):
        z = self.backbone_momentum(x)
        if self.projection_head_momentum is not None:
            z = self.projection_head_momentum(z)
        return z.detach()

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        x = batch[0]
        p, _, target_mask_b1ff = self.forward(x)
        h = self.forward_momentum(x)
        # Normalize in feature dimension separately for each patch
        p = F.normalize(p, dim=1)
        h = F.normalize(h, dim=1)
        loss = F.smooth_l1_loss(p, h, reduction='none').sum(axis=1,keepdim=True) # (B, 1, H, W)
        # loss = F.cosine_similarity(p, h, dim=1).unsqueeze(1)  # (B, 1, H, W)
        neg_mask_b1ff = target_mask_b1ff
        loss = loss.mul_(neg_mask_b1ff).sum() / (neg_mask_b1ff.sum() + 1e-8)  # loss only on masked patches
        self.log(f"{metric_label}/ijepa_loss", loss, on_epoch=True)
        return loss
    
    # def configure_optimizers(self):
    #     # Don't use weight decay for batch norm, bias parameters, and classification
    #     # head to improve performance.
    #     params, params_no_weight_decay = get_weight_decay_parameters(
    #         [
    #             self.backbone_sparse,
    #             self.predictor,
    #         ] + 
    #         ([self.projection_head] if self.projection_head is not None else [])
    #     )
    #     param_groups = [
    #             {"name": "model", "params": params},
    #             {
    #                 "name": "model_no_weight_decay",
    #                 "params": params_no_weight_decay,
    #                 "weight_decay": 0.0,
    #             },
    #     ]
    #     # optimizer = torch.optim.AdamW(
    #     #     param_groups,
    #     #     lr=self.lr,
    #     #     weight_decay=self.cfg.optimizer.weight_decay,
    #     # )
    #     optimizer = LARS(
    #         param_groups,
    #         lr=self.cfg.optimizer.lr * self.cfg.optimizer.batch_size * self.trainer.world_size / 256,
    #         momentum=0.9,
    #         weight_decay=self.cfg.optimizer.weight_decay,
    #     )
    #     scheduler = {
    #         "scheduler": CosineWarmupScheduler(
    #             optimizer=optimizer,
    #             warmup_epochs=int(
    #                 self.trainer.estimated_stepping_batches
    #                 / self.trainer.max_epochs
    #                 * 10
    #             ),
    #             max_epochs=int(self.trainer.estimated_stepping_batches),
    #         ),
    #         "interval": "step",
    #     }
    #     return [optimizer], [scheduler]


@hydra.main(version_base="1.2", config_path="configs/", config_name="ijepacnn.yaml")
def pretrain_byol(cfg: DictConfig):
    main_pretrain(cfg, IJEPA_CNN)

if __name__ == "__main__":
    pretrain_byol()
