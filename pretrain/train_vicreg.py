# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from omegaconf import DictConfig
import torch
from torch import nn
import torch.nn.functional as F

from lightly.loss import VICRegLoss
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.transforms.vicreg_transform import VICRegTransform

from pretrain.trainer_common import LightlyModel, main_pretrain

class VICReg(LightlyModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if self.cfg.data.dataset_name in ["cifar10", "stl10"]:
            self.projection_head = VICRegProjectionHead(
                input_dim=self.backbone.num_features,
                hidden_dim=2048,
                output_dim=2048,
                num_layers=2,
                norm_layer=self.projector_norm
            )
        else:
            self.projection_head = VICRegProjectionHead(self.backbone.num_features)
        self.criterion = VICRegLoss()

    def setup_transform(self):
        self.transform = VICRegTransform(input_size=self.input_size)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return z

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log(f"{metric_label}/vicreg_loss", loss, on_epoch=True)
        return loss

@hydra.main(version_base="1.2", config_path="configs/", config_name="vicreg.yaml")
def pretrain_vicreg(cfg: DictConfig):
    main_pretrain(cfg, VICReg)

if __name__ == "__main__":
    pretrain_vicreg()