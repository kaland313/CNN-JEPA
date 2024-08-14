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
from torchvision import transforms
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import knn_predict
from lightly.utils.benchmarking.topk import mean_topk_accuracy

from pretrain.trainer_common import LightlyModel, main_pretrain

class Supervised(LightlyModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.topk = (1, 5)

    def setup_transform(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"],
                    std=IMAGENET_NORMALIZE["std"],
                ),
            ]
        )
        # Slight HACK here: override the configuration for the train dataset for stl10 from unlabeled+train to train
        if self.cfg.data.dataset_name == "stl10":
            self.train_dataset_kwargs['split'] = 'train'

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self.classifier = nn.Linear(self.backbone.num_features, self.num_classes)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        y = self.classifier(y)
        return y

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        x, label, _ = batch
        y = self.forward(x)
        loss = self.criterion(y, label)

        _, predicted_classes = y.topk(max(self.topk))

        topk = mean_topk_accuracy(
            predicted_classes=predicted_classes, targets=label, k=self.topk
        )
        results_dict = {f"{metric_label}/supervised_acc_top{k}": acc for k, acc in topk.items()}
        self.log(f"{metric_label}/supervised_loss", loss, on_epoch=True)
        self.log_dict(results_dict, on_epoch=True)
        return loss

    def get_views_to_log_from_batch(self, batch):
        # a batch in lightly is a tuple: inputs, targets, filepaths. Views are in batch[0]
        inputs, targets, filepaths = batch
        return [inputs]

@hydra.main(version_base="1.2", config_path="configs/", config_name="supervised.yaml")
def pretrain_supervised(cfg: DictConfig):
    main_pretrain(cfg, Supervised)

if __name__ == "__main__":
    pretrain_supervised()