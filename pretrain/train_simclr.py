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

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

from pretrain.trainer_common import LightlyModel, main_pretrain


class SimCLR(LightlyModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.projection_head = SimCLRProjectionHead(
            input_dim=self.backbone.num_features
        )
        self.criterion = NTXentLoss(temperature=0.1, gather_distributed=True)

    def setup_transform(self):
        self.transform = SimCLRTransform(input_size=self.input_size)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return z

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        x0, x1 = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log(f"{metric_label}/simclr_loss", loss, on_epoch=True)
        return loss

@hydra.main(version_base="1.2", config_path="configs/", config_name="simclr.yaml")
def pretrain_simclr(cfg: DictConfig):
    main_pretrain(cfg, SimCLR)

if __name__ == "__main__":
    pretrain_simclr()