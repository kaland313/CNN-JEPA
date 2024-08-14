# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig, OmegaConf
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import knn_predict
from lightly.utils.benchmarking.topk import mean_topk_accuracy
import pytorch_lightning as pl

class OnlineLinearClassificationBenckmark:
    def __init__(
        self,
        backbone,
        num_classes,
        dataset_class,
        train_dataset_kwargs,
        val_dataset_kwargs,
        input_size,
        batch_size=256,
        num_workers=8,
        topk=(1,5),
        dist_world_size=1,
        dist_rank=0,
    ):
        self.backbone = backbone
        self.num_features = backbone.num_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.topk = topk
        self.dist_world_size = dist_world_size
        self.dist_rank = dist_rank

        # Dataset & Dataloader setup
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"],
                    std=IMAGENET_NORMALIZE["std"],
                ),
            ]
        )

    def run_benchmarks(self,
                       device,
                       dist_all_gather_fcn,
                       train_dataloader,
                       val_dataloader,
                       train_val_transform,
                       num_epochs=90, ):
        """ Runs the online linear classification benchmark.

        Args:
            num_epochs (int, optional): Trains the linear classification layer for num_epochs epochs. Defaults to 2.

        Returns:
            dict: benchmark results
        """
        # train_dataloader.dataset is expected to be pytorch_lightning.trainer.supporters.CombinedDataset
        # train_dataloader.dataset.datasets is expected to be a LightlyDataset
        # We replace the dataset's transform with the one we want to use for the benchmark
        # and restore it at the end
        # If we do this with another transform attribute we might run multiple transforms or the wrong one.
        assert train_dataloader.dataset.datasets.transform == train_val_transform, "train_dataloader.dataset.datasets.transform != train_val_transform, adapt this and the lines where we change the transform to the dataset"
        train_dataloader.dataset.datasets.transform = self.transform

        # val_dataloader.dataset is expected to be a LightlyDataset
        assert val_dataloader.dataset.transform == train_val_transform, "val_dataloader.dataset.datasets.transform != train_val_transform, adapt this and the lines where we change the transform to the dataset"
        val_dataloader.dataset.transform = self.transform

        train_features, train_targets = self.compute_features(train_dataloader, device)
        val_features, val_targets = self.compute_features(val_dataloader, device)

        train_features = dist_all_gather_fcn(train_features)
        train_targets = dist_all_gather_fcn(train_targets)
        val_features = dist_all_gather_fcn(val_features)
        val_targets = dist_all_gather_fcn(val_targets)
        if train_features[0].dim() > 2:
            train_features = [f.flatten(0,1) for f in train_features]
            train_targets = [t.flatten(0,1) for t in train_targets]
            val_features = [f.flatten(0,1) for f in val_features]
            val_targets = [t.flatten(0,1) for t in val_targets]

        if self.dist_rank == 0:
            # Initialize the linear classifier
            classifier = nn.Linear(self.num_features, self.num_classes).to(device)
            for p in classifier.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(classifier.parameters())

            lin_accs = None
            for epoch in range(num_epochs):
                self.fit_lin_classifier(train_features, train_targets, classifier, optimizer)
                lin_results_dict = self.evaluate_lin_classifier(val_features, val_targets, classifier)
                if lin_accs is None:
                    lin_accs = {k: [v] for k, v in lin_results_dict.items()}
                else:
                    for k, v in lin_results_dict.items():
                        lin_accs[k].append(v)
                print(f"\nBenchmark Epoch {epoch+1}/{num_epochs}, Lin Accuracy: ", *[f"{k}: {v*100:.2f}%" for k, v in lin_results_dict.items()])
                # rename final lin_results_dict items by appending _final
                lin_results_dict = {f"{k}_final": v for k, v in lin_results_dict.items()}
                # compute max for each lin_acc and store it in lin_results_dict
                lin_results_dict.update({k: max(v) for k, v in lin_accs.items()})                

            try:
                knn_results_dict = self.evaluate_knn_classifier(
                    train_features, train_targets, val_features, val_targets
                )
            except RuntimeError as e:
                print(f"\n Error in KNN evaluation: {e}\n")
                knn_results_dict = {}

            results_dict = {**lin_results_dict, **knn_results_dict}
            print(f"\nAccuracy: ", *[f"\n  {k}: {v*100:.2f}%" for k, v in results_dict.items()], "\n\n")

            # cleanup
            del classifier, optimizer
        else:
            results_dict = {}

        # cleanup
        del train_features, train_targets, val_features, val_targets
        train_dataloader.dataset.datasets.transform = train_val_transform
        val_dataloader.dataset.transform = train_val_transform
        return results_dict
    
    @torch.no_grad()
    def compute_features(self, dataloader, device):
        """
        Compute features for the whole dataset.
        """
        features = []
        targets = []
        for batch in tqdm(dataloader, desc=f"Computing features on {device}"):
            inputs, targets_batch = batch[0], batch[1]
            inputs = inputs.to(device)
            targets_batch = targets_batch.to(device)

            representations = self.backbone(inputs)
            if len(representations.shape) > 2:
                # if we get pre-pooling feature maps, pool them.
                representations = torch.flatten(
                    F.adaptive_avg_pool2d(representations, 1), start_dim=1
                )
            features.append(representations)
            targets.append(targets_batch)
        return features, targets
    
    @staticmethod
    def fit_lin_classifier(train_features, train_targets, classifier, optimizer):
        for batch in zip(train_features, train_targets):
            features_batch, targets_batch = batch
            # Classifier forward pass and optimization
            with torch.enable_grad():
                # If we call online_linear_classification_benchmark from a lightning module's on_validation_epoch_end, 
                # gradient computation is disabled by default. (check it with torch.is_grad_enabled())
                # For training the linear classifier we need to enable it again.
                optimizer.zero_grad()
                outputs = classifier(features_batch)
                loss = nn.CrossEntropyLoss()(outputs, targets_batch)
                loss.backward()
                optimizer.step()
    
    @torch.no_grad()
    def evaluate_lin_classifier(self, val_features, val_labels, classifier):
        val_features_tensor = torch.cat(val_features, dim=0)
        val_labels_tensor = torch.cat(val_labels, dim=0)

        outputs = classifier(val_features_tensor)
        
        _, predicted_classes = outputs.topk(max(self.topk))

        topk = mean_topk_accuracy(
            predicted_classes=predicted_classes, targets=val_labels_tensor, k=self.topk
        )
        results_dict = {f"lin_top{k}": acc for k, acc in topk.items()}
        return results_dict
    
    @torch.no_grad()
    def evaluate_knn_classifier(self, feature_bank, label_bank, val_features, val_labels, k=200, t=0.1, eval_batch_size=64):
        feature_bank_tensor = torch.cat(feature_bank, dim=0)
        label_bank_tensor = torch.cat(label_bank, dim=0)
        val_labels_tensor = torch.cat(val_labels, dim=0)

        # Rebatch the validation features to avoid OOM
        val_features_rebatched = []
        for f in val_features:
            val_features_rebatched.extend(torch.split(f, eval_batch_size, dim=0))
        val_features = val_features_rebatched

        feature_bank_tensor = F.normalize(feature_bank_tensor, dim=1).T

        predicted_classes = []

        for val_features_tensor in tqdm(val_features, desc="Evaluating kNN"):
            val_features_tensor = F.normalize(val_features_tensor, dim=1)

            predicted_classes_batch = knn_predict(
                feature=val_features_tensor,
                feature_bank=feature_bank_tensor,
                feature_labels=label_bank_tensor,
                num_classes=self.num_classes,
                knn_k=k,
                knn_t=t,
            )
            predicted_classes.append(predicted_classes_batch)
        
        predicted_classes = torch.cat(predicted_classes, dim=0)

        topk = mean_topk_accuracy(
            predicted_classes=predicted_classes, targets=val_labels_tensor, k=self.topk
        )
        results_dict = {f"knn_top{k}": acc for k, acc in topk.items()}

        return results_dict
        
    @torch.no_grad()
    def compute_dummy_features(self, dataloader, device):
        """Use:
            model.online_classifier.compute_features = model.online_classifier.compute_dummy_features
        """
        num_features = self.backbone.num_features
        features = [
            torch.randn(self.batch_size, num_features).to(device) for _ in range(len(dataloader.dataset) // self.batch_size)
        ]
        targets = [
            torch.randint(0, self.num_classes, (self.batch_size,)).to(device) for _ in range(len(dataloader.dataset)  // self.batch_size)
        ]
        return features, targets


def test_online_classification():
    from pretrain.trainer_common import LightlyModel

    cfg = DictConfig(
        {
            "data": {
                "dataset_name": "imagenette",
                "num_workers": 8,
            },
            "backbone": {
                "name": "resnet50",
                "pretrained_weights": "imagenet",
                "kwargs": {},
            },
            "optimizer": {
                "lr": None,
            },
        }
    )

    model = LightlyModel(cfg)
    model.setup("validate")
    model.online_classifier.device = "cuda"
    model.backbone.to(model.online_classifier.device)
    # model.online_classifier.compute_features = model.online_classifier.compute_dummy_features
    
    model.online_classifier.run_benchmarks("cuda", lambda x: x)

def test_online_classification_2():
    len_train_ds = 1281167
    len_val_ds = 50000
    num_classes = 1000
    num_features = 2048
    batch_size = 256
    device = "cuda"
    dummy_train_features = [torch.randn(batch_size, num_features).to(device) for _ in range(len_train_ds // batch_size)]
    dummy_train_labels = [torch.randint(0, num_classes, (batch_size,)).to(device) for _ in range(len_train_ds // batch_size)]
    dummy_val_features = [torch.randn(batch_size, num_features).to(device) for _ in range(len_val_ds // batch_size)]
    dummy_val_labels = [torch.randint(0, num_classes, (batch_size,)).to(device) for _ in range(len_val_ds // batch_size)]

    class DummyEvaluator():
        num_classes = 1000
        topk = (1, 5)

    DummyEvaluator.evaluate_knn_classifier = OnlineLinearClassificationBenckmark.evaluate_knn_classifier

    evaluator = DummyEvaluator()
    knn_results = evaluator.evaluate_knn_classifier(dummy_train_features, dummy_train_labels, dummy_val_features, dummy_val_labels)
    print(knn_results) 


def eval_model(lightly_model):
    lightly_model.lr = 0.0

    # Create a trainer and run training for 0 steps
    world_size = os.environ.get("SLURM_NTASKS")
    if world_size is not None:
        world_size = int(world_size)
    print("World size:", world_size, flush=True)
    if world_size == 1:
        strategy = None
    else:
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        logger=False, 
        strategy=strategy, 
        devices="auto",
        num_nodes=os.environ.get("SLURM_NNODES"),
        accelerator="gpu",
        precision="bf16",
        max_steps=1,
        limit_val_batches=0.1,
        val_check_interval=0.0000001,
        enable_checkpointing=False,
    )
    trainer.fit(lightly_model)

def eval_ckpt(ckpt_path):
    from pretrain.train_ijepacnn import IJEPA_CNN
    from pretrain.train_byol import BYOL

    # load a LightlyModel from a checkpoint
    if "byol" in ckpt_path:
        model = BYOL.load_from_checkpoint(ckpt_path)
    else:
        model = IJEPA_CNN.load_from_checkpoint(ckpt_path)
    
    eval_model(model)


def eval_lightly_benchmark_ckpt(ckpt_path="lightly_byol.ckpt"):
    from pretrain.train_byol import BYOL

    cfg = DictConfig(
        {
            "data": {
                "dataset_name": "imagenet-1k",
                "num_workers": 8,
            },
            "backbone": {
                "name": "resnet50",
                "pretrained_weights": "imagenet",
                "kwargs": {},
            },
            "optimizer": {
                "lr": None,
                'weight_decay': 0.0,
                'batch_size': 128,
            },
            "trainer": {
                "max_epochs": 101,
                "devices": "auto",
                "accelerator": "gpu",
                "precision": "bf16",
                "sync_batchnorm": True,
            }
        }
    )

    model = BYOL(cfg)

    ckpt = torch.load(ckpt_path)
    ckpt = ckpt["state_dict"]
    # drop non-backbone keys
    ckpt = {k.replace("backbone.", ""): v for k, v in ckpt.items() if k.startswith("backbone.")}
    model.backbone.load_state_dict(ckpt)
    
    eval_model(model)

if __name__ == "__main__":
    # test_online_classification()

    # eval_ckpt("artifacts/pretrain_lightly/ijepacnn_imagenet-1k/I-JEPA_imagenet-1k_resnet50_predL3K9_Mixed_lr0.01_wd0.01_bs128_predblocks4_4GPU/version_0/epoch=100-step=252702.ckpt")

    eval_lightly_benchmark_ckpt("lightly_simclr.ckpt")
