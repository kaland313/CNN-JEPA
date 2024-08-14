# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import os
import copy

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import timm

from lightly.data import LightlyDataset
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler

from pretrain.metrics import compute_contrastive_acc, log_example_inputs
from pretrain.trainer_common import LightlyModelMomentum, main_pretrain


class BYOL(LightlyModelMomentum):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if self.cfg.data.dataset_name == "imagenet-1k":
            self.projection_head = BYOLProjectionHead(self.backbone.num_features)
            self.prediction_head = BYOLPredictionHead()
        else:
            self.projection_head = BYOLProjectionHead(self.backbone.num_features, 1024, 256)
            self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def setup_transform(self):
        self.transform = BYOLTransform(
            view_1_transform=BYOLView1Transform(input_size=self.input_size),
            view_2_transform=BYOLView2Transform(input_size=self.input_size)
            )

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        x0, x1 = batch[0]
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(f"{metric_label}/byol_loss", loss, on_epoch=True)
        return loss
    
    # def configure_optimizers(self):
    #     # Don't use weight decay for batch norm, bias parameters, and classification
    #     # head to improve performance.
    #     params, params_no_weight_decay = get_weight_decay_parameters(
    #         [
    #             self.backbone,
    #             self.projection_head,
    #             self.prediction_head,
    #         ]
    #     )
    #     optimizer = LARS(
    #         [
    #             {"name": "byol", "params": params},
    #             {
    #                 "name": "byol_no_weight_decay",
    #                 "params": params_no_weight_decay,
    #                 "weight_decay": 0.0,
    #             },
    #         ],
    #         # Settings follow original code for 100 epochs which are slightly different
    #         # from the paper, see:
    #         # https://github.com/deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/byol/configs/byol.py#L21-L23
    #         lr=self.cfg.optimizer.lr * self.cfg.optimizer.batch_size * self.trainer.world_size / 256,
    #         momentum=0.9,
    #         weight_decay=1e-6,
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


@hydra.main(version_base="1.2", config_path="configs/", config_name="byol.yaml")
def pretrain_byol(cfg: DictConfig):
    main_pretrain(cfg, BYOL)

if __name__ == "__main__":
    pretrain_byol()
