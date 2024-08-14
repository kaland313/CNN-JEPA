# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import time
from typing import Callable
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from lightning_lite.utilities.rank_zero import _get_rank
import torch
import torchvision
import torch.nn as nn
import timm
import wandb
import matplotlib.pyplot as plt

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule, CosineWarmupScheduler
from lightly.utils.lars import LARS

from pretrain.metrics import contrastive_acc_eval, log_example_inputs, eval_feature_descriptors
from pretrain.online_classification_benchmark import OnlineLinearClassificationBenckmark
import utils

from data.imagenette import Imagenette
from data.cached_imagenet import CachedImageNet
from data.hdf5_imagefolder import HDF5ImageFolder

class LightlyModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()  # save cfg to self.hparams
        self.cfg = cfg
        self.lr = cfg.optimizer.lr

        self.backbone = timm.create_model(
            cfg.backbone.name,
            pretrained=cfg.backbone.pretrained_weights == "imagenet",
            num_classes=0,
            **dict(cfg.backbone.kwargs),
        )
        # This sets which backbone to use for online eval. By default, it's the same as the main backbone
        # Override this if needed (e.g. with backbone_momentum)
        self.backbone_for_online_eval = self.backbone 

        self.projection_head = None
        self.criterion = None

    def forward(self, x):
        """Implment forward step for each method!

        Args:
            x: a minibatch of augmented input images
        """
        raise NotImplemented

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        """Implment train_val step for each method!
        log loss using self.log(f"{metric_label}/loss", loss, on_epoch=True)
        """
        raise NotImplemented

    def setup_transform(self):
        """ Set sef.transform to a lightly transform by oveerriding this method.
            Use sef.input_size to set the input_size of the transform.
        """
        # We set self.transform to an invalid value to allow this function to be called, but if it's not overriden, we raise an error
        # self.setup calls this method, therefore this hack to allows this class to be instantiated without having to override this method
        self.transform = -1

    def training_step(self, batch, batch_idx):
        loss = self.train_val_step(batch, batch_idx)
        self.log(f"train_metrics/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log(f"train_metrics/wd", self.trainer.optimizers[0].param_groups[0]["weight_decay"])
        
        if hasattr(self, "wd_scheduler"):
            self.wd_scheduler.step()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.train_val_step(batch, batch_idx, metric_label="val_metrics")
        if self.trainer.sanity_checking:
            views = self.get_views_to_log_from_batch(batch)
            shuffle = torch.randperm(views[0].shape[0])
            views = [view[shuffle] for view in views]
            log_example_inputs(views, log_label="val")
        return loss

    def get_views_to_log_from_batch(self, batch):
        # a batch in lightly is a tuple: inputs, targets, filepaths. Views are in batch[0]
        # Override this if the transforms doewsn't return multiple views in inputs
        return batch[0]

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            if self.current_epoch % 5 == 0:
                try:
                    benchmark_results_dict = self.online_classifier.run_benchmarks(
                        device=self.device,
                        dist_all_gather_fcn=self.all_gather,
                        train_dataloader=self.trainer.train_dataloader,
                        val_dataloader=self.trainer.val_dataloaders[0],
                        train_val_transform=self.transform,
                    )
                except Exception as e:
                    print(f"Failed to run online classification benchmarks: {e}", flush=True)
                    benchmark_results_dict = {"lin_top1": 0.0}

                # if benchmark_results_dict is not None:
                self.log_dict({f"val_metrics/{k}": v for k, v in benchmark_results_dict.items()})
                # https://github.com/Lightning-AI/pytorch-lightning/issues/19045

    def configure_optimizers(self):
        if self.cfg.optimizer.get('exclude_norm_and_bias_from_wd', False):
            params, params_no_weight_decay, _, param_names_no_weight_decay = utils.get_weight_decay_parameters(self.named_parameters())
            print("Parameters excluded from weight decay:", param_names_no_weight_decay, flush=True)
            param_groups = [
                {
                    'params': params
                }, 
                {
                    'params': params_no_weight_decay,
                    'WD_exclude': True, # important for CosineWDSchedule
                    'weight_decay': 0
                }
            ]
        else:
            param_groups = self.parameters()

        wd = self.cfg.optimizer.get('weight_decay', 0.0)
        if self.cfg.optimizer.get('algorithm', 'adamw').lower() == 'lars':
            optim = LARS(
                param_groups,
                lr=self.cfg.optimizer.lr,
                momentum=0.9,
                weight_decay=wd,
            )
        else:        
            optim = torch.optim.AdamW(
                param_groups,
                lr=self.lr,
                weight_decay=wd,
            )

        if self.cfg.optimizer.get('wd_schedule', False): 
            self.wd_scheduler = utils.CosineWDSchedule(
                optim,
                ref_wd=wd,
                final_wd=wd * 10,
                T_max=int(self.trainer.estimated_stepping_batches),
            )

        if self.cfg.optimizer.get('cosine_warmpup_sched', False):
            scheduler = {
                "scheduler": CosineWarmupScheduler(
                    optimizer=optim,
                    warmup_epochs=int(
                        self.trainer.estimated_stepping_batches
                        / self.trainer.max_epochs
                        * self.cfg.optimizer.get('lr_warmup_epochs', 10)
                    ),
                    max_epochs=int(self.trainer.estimated_stepping_batches),
                ),
                "interval": "step",
            }
            return [optim], [scheduler]
        else:
            return optim

    def setup(self, stage: str) -> None:
        dataset_classes = {
            "cifar10": torchvision.datasets.CIFAR10,
            "stl10": torchvision.datasets.STL10,
            "tiny-imagenet": torchvision.datasets.ImageFolder,
            "imagenette": Imagenette,
            "imagenet-100": HDF5ImageFolder, # Replaceable with torchvision.datasets.ImageFolder
            "imagenet-1k":  HDF5ImageFolder, # Replaceable with torchvision.datasets.ImageFolder
        }
        train_dataset_kwargs = {
            "cifar10": dict(root="/data/cifar10", download=True),
            "stl10": dict(root="/data/stl10", download=True, split='train+unlabeled'),
            "tiny-imagenet": dict(root="/data/tiny-imagenet-200/train"),
            "imagenette": dict(root="/data/imagenette", split='train', download=True),
            "imagenet-100": dict(root="/data/imagenet-100-train.h5"),
            "imagenet-1k": dict(root="/data/imagenet-train.h5"),
        }
        val_dataset_kwargs = {
            "cifar10": dict(root="/data/cifar10", train=False),
            "stl10": dict(root="/data/stl10", split='test'),
            "tiny-imagenet": dict(root="/data/tiny-imagenet-200/val"),
            "imagenette": dict(root="/data/imagenette", split='val'),
            "imagenet-100": dict(root="/data/imagenet-100-val.h5"),
            "imagenet-1k": dict(root="/data/imagenet-val.h5"),
        }
        input_sizes = {
            "cifar10": 32,
            "stl10":  96,
            "tiny-imagenet": 64,
            "imagenette": 224,
            "imagenet-100": 224,
            "imagenet-1k": 224,
        }
        num_classes = {
            "cifar10": 10,
            "stl10":  10,
            "tiny-imagenet": 200,
            "imagenette": 10,
            "imagenet-100": 100,
            "imagenet-1k": 1000,
        }
        self.dataset_class = dataset_classes[self.cfg.data.dataset_name]
        self.train_dataset_kwargs = train_dataset_kwargs[self.cfg.data.dataset_name]
        self.val_dataset_kwargs = val_dataset_kwargs[self.cfg.data.dataset_name]
        self.input_size = input_sizes[self.cfg.data.dataset_name]
        self.num_classes = num_classes[self.cfg.data.dataset_name]

        # Setup self.transform
        self.setup_transform()

        self.train_dataset = LightlyDataset.from_torch_dataset(
            self.dataset_class(**self.train_dataset_kwargs),
            transform=self.transform
        )
        self.val_dataset = LightlyDataset.from_torch_dataset(
            self.dataset_class(**self.val_dataset_kwargs),
            transform=self.transform
        )

        lin_benchmark_train_kwargs = self.train_dataset_kwargs.copy()
        if self.cfg.data.dataset_name == "stl10":
            lin_benchmark_train_kwargs["split"] = "train"
        self.online_classifier = OnlineLinearClassificationBenckmark(
            backbone=self.backbone_for_online_eval,
            num_classes=self.num_classes, 
            dataset_class=self.dataset_class,
            train_dataset_kwargs = lin_benchmark_train_kwargs, 
            val_dataset_kwargs=self.val_dataset_kwargs, 
            input_size=self.input_size,
            num_workers=self.cfg.data.num_workers,
            dist_world_size=self.trainer.world_size if self._trainer is not None else 1,
            dist_rank=self.trainer.global_rank if self._trainer is not None else 0,
        ) # WARNING: At this point device is CPU!!!

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.optimizer.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.data.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.optimizer.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
        )
        return dataloader


class LightlyModelMomentum(LightlyModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        deactivate_requires_grad(self.backbone_momentum)

        self.projection_head_momentum = None

    def forward_momentum(self, x):
        raise NotImplemented
    
    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, self.cfg.trainer.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        if self.projection_head_momentum is not None:
            update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        return super().training_step(batch, batch_idx)


def main_pretrain(cfg: DictConfig, lightly_model: LightlyModel):
    print("Running on:", os.environ.get("HOSTNAME", "docker"), flush=True)
    os.system("nvidia-smi")
    print(torch.cuda.device_count(), "GPUs available", flush=True)

    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    OmegaConf.set_struct(cfg, False)

    cfg.artifacts_root += "_" + cfg.data.dataset_name

    flat_config = utils.flatten_dict(cfg)
    cfg.name = cfg.name.format(**flat_config)

    pl.seed_everything(cfg.seed)

    model = lightly_model(cfg)

    if cfg.wandb:
        wandb_logger = pl.loggers.WandbLogger(
            name=cfg.name, project="I-JEPA-CNN", save_dir="artifacts",
            group=cfg.get("wandb_group", None),
        )
        # wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    root_dir = os.path.abspath(os.path.join(cfg.artifacts_root, cfg.name))
    version = utils.get_next_version(root_dir)
    ckpt_dir = os.path.join(root_dir, f"version_{version}")
    time.sleep(3) # To allow for other ranks to get the version number right
    if _get_rank() == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    print("Checkpoint dir:", ckpt_dir, flush=True)
    
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,  # False to reduce disk load from constant checkpointing
        save_on_train_epoch_end=True,
        # save_top_k=1, # Doesn't work with DDP 
        # monitor="val_metrics/lin_top1",
        # mode="max",
    )
    callbacks = [checkpoint]

    # Note:
    # - DDP find_unused_parameters=False set because: https://pytorch-lightning.readthedocs.io/en/1.8.6/advanced/model_parallel.html?highlight=find_unused_parameter
    # - DDPStrategy vs DDPSpawnStrategy: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#distributed-data-parallel-spawn
    world_size = os.environ.get("SLURM_NTASKS")
    if world_size is not None:
        world_size = int(world_size)
    print("World size:", world_size, flush=True)
    if world_size == 1:
        strategy = None
    else:
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        logger=[wandb_logger] if cfg.wandb else False, 
        callbacks=callbacks, 
        strategy=strategy, 
        num_nodes=os.environ.get("SLURM_NNODES") or 1, # if SLURM_NNODES is not set, we assume 1 node
        **cfg.trainer,
    )
    trainer.fit(model=model)

if __name__ == "__main__":
    main_pretrain(LightlyModel)
