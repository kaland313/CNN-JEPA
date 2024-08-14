# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from tqdm import tqdm
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

from lightly.transforms import SimCLRTransform
from lightly.data import LightlyDataset
# from lightly.transforms.utils import IMAGENET_NORMALIZE
IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

def compute_contrastive_acc(features_0, features_1):
    # Based on https://github.com/beresandras/contrastive-classification-keras/blob/c32077d073c74cd741a456415035a77b5db1df77/models.py#L42
    # Compute pairwise cosine similarity between the feature vectors of each sample
    # Features_0 and features_1 should contrasitve paired features, i.e. feature_0[i] and feature_1[i]
    # should come from 2 different augmentations of the same image.
    # features_0, features_1 shape (batch_size, embedding_dim)
    features_0 = torch.nn.functional.normalize(features_0.detach(), dim=1)
    features_1 = torch.nn.functional.normalize(features_1.detach(), dim=1)
    # similarities[i,i] show how similar are the feature vectors of the two augmentations of one image
    # similarities[i,j] show how similar are the feature vector of different images
    similarities = torch.matmul(features_0, features_1.T)  # shape (batch_size, batch_size)

    # To each sample in a minibatch we assign the sample with the highest similarity
    contrastive_preds = torch.argmax(torch.concat([similarities, similarities.T], dim=0), dim=1)

    batch_size = features_0.shape[0]
    contrastive_labels = torch.concat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)

    contrastive_acc = (
        torch.sum(contrastive_preds.cpu() == contrastive_labels) / contrastive_preds.shape[0]
    )
    return contrastive_acc


def contrastive_acc_eval(backbone: nn.Module, dataset, batch_size: int = 64, input_size: int = 224):
        contrastive_acc_transform = SimCLRTransform(input_size=input_size)

        dataset = LightlyDataset.from_torch_dataset(dataset, transform=contrastive_acc_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=8,
        )
        contrastive_accs = []
        backbone.eval()
        for batch in tqdm(dataloader, desc="Computing contrastive accuracy"):
            x0, x1 = batch[0]
            x0 = x0.to("cuda")
            x1 = x1.to("cuda")
            z0 = backbone(x0)
            z1 = backbone(x1)
            if len(z0.shape) > 2:
                # if we get pre-pooling feature maps, pool them. 
                # compute_contrastive_acc expects (batch,features) shaped tensors
                z0 = torch.flatten(F.adaptive_avg_pool2d(z0, 1), start_dim=1)
                z1 = torch.flatten(F.adaptive_avg_pool2d(z1, 1), start_dim=1)
            contrastive_acc = compute_contrastive_acc(z0, z1)
            contrastive_accs.append(contrastive_acc)
            
        contrastive_acc = torch.mean(torch.stack(contrastive_accs))

        return contrastive_acc


def log_example_inputs(views, log_label="train", num_examples=8):
    batch_size = views[0].shape[0]
    num_examples = min(num_examples, batch_size)
    selected_views = [view[:num_examples].cpu().numpy() for view in views]
    max_size = np.max([view.shape[-1] for view in selected_views], axis=0)
    selected_views = [
        np.pad(view, ((0, 0), (0, 0), (0, max_size - view.shape[2]), (0, 0)))
        for view in selected_views
    ]
    imgs = np.concatenate(selected_views, axis=-1).transpose(0, 2, 3, 1)
    wandb_imgs = [wandb.Image(imgs[i]) for i in range(num_examples)]
    if wandb.run is not None:
        wandb.log({f"examples/{log_label}": wandb_imgs})
        # log x0 min and max, mean
        wandb.log(
            {
                f"examples/{log_label}_x0_min": views[0].min(),
                f"examples/{log_label}_x0_max": views[0].max(),
                f"examples/{log_label}_x0_mean": views[0].mean(),
            }
        )

@torch.no_grad()
def eval_feature_descriptors(backbone: nn.Module, dataset, batch_size: int = 64, input_size: int = 224, cfg_name=None, current_epoch=None):
    """ Compute feature descriptors for a dataset. 
        
    Args:
        backbone (nn.Module): The backbone model
        dataset (torch.utils.data.Dataset): The dataset to compute feature descriptors for
        batch_size (int, optional): The batch size to use. Defaults to 64.
        input_size (int, optional): The input size of the images. Defaults to 224.
        cfg_name ([type], optional): The name of the experiment used in logging, e.g. cfg.name . Defaults to None.
        current_epoch ([type], optional): The current epoch for logging feature tensors. Defaults to None.
    """
    minimal_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=IMAGENET_NORMALIZE['mean'],
                                         std=IMAGENET_NORMALIZE['std']),
    ])
    dataset.transform = minimal_transforms
    if batch_size == -1:
        batch_size = len(dataset)     
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
    )
    feats = []
    labels = []
    backbone.eval()
    for batch in tqdm(dataloader, desc="Computing feature descriptors"):
        x0, label = batch
        x0 = x0.to("cuda")
        z0 = backbone(x0)
        if len(z0.shape) > 2:
            # if we get pre-pooling feature maps, pool them. 
            # compute_contrastive_acc expects (batch,features) shaped tensors
            z0 = torch.flatten(F.adaptive_avg_pool2d(z0, 1), start_dim=1)
        feats.append(z0.detach())
        labels.append(label.detach()) 

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    feature_descriptors = compute_feature_descriptors(feats)

    # if the number of features == 2, concatenate them and create a simple 2d scatter plot
    if z0.shape[1] == 2:
        feats = feats.cpu().numpy()
        labels = labels.cpu().numpy()
        fig = plt.scatter(feats[:, 0], feats[:, 1], c=labels, cmap="tab10")
        wandb.log({f"val_metrics/2d_feature_scatterplot": wandb.Image(fig)})

        fig_path = os.path.join("feature_scatterplot_2d", cfg_name)
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(f"{fig_path}.png", dpi=600)
        plt.close()
        # save features, labels to a numpy file {fig_path}/f"{current_epoch}.npz
        np.savez(os.path.join(fig_path, f"{current_epoch}.npz"), feats=feats, labels=labels)

    return feature_descriptors


def compute_feature_descriptors(z):
    """ Compute feature descriptors for a batch of features z.
    The descriptors are:
    - corr: mean correlation between feature dimensions (mean of off-diagonal elements of the correlation matrix)
    - rank: rank of the feature matrix
    - std: mean standard deviation of the feature dimensions
    
    Based on: https://github.com/PatrickHua/FeatureDecorrelationSSL/blob/main/models/covnorm.py

    Args:
        z (_type_): Features provided by the backbone, shape (batch_size, embedding_dim)
    """
    # corr = torch.corrcoef(z.detach().T).abs() # gives the same result as corrcoef defined below
    corr = corrcoef(z.detach()).abs()
    D = corr.shape[0]
    mean_corr = corr.fill_diagonal_(0).sum() / (D*(D-1))

    if torch.isnan(mean_corr):
        mean_corr = torch.tensor(-1.)
        ns_mean_corr = corr.fill_diagonal_(0).nansum() / (D*(D-1))
    else:
        ns_mean_corr = mean_corr

    tol = 1e-1
    try:
        rank = torch.matrix_rank(z.detach().cpu(), tol=tol)
        rank_ratio = rank / min(z.shape)
    except Exception:
        rank = torch.tensor(-1.)
        rank_ratio = torch.tensor(-1.)

    try:
        std = z.detach().std(dim=0).mean()
    except Exception:
        std = torch.tensor(-1.)

    # anisotropy
    s = torch.linalg.svdvals(z.detach().cpu())
    anisotropy = s[0]**2 / (s**2).sum()

    return {'corr': mean_corr.cpu(), 
            'nansafe_corr': ns_mean_corr.cpu(), 
            'rank': rank.to(torch.float32), 
            'rank_ratio': rank_ratio.to(torch.float32),
            'std': std.cpu(),
            'anisotropy': anisotropy}

def covariance(x):
    x = x - x.mean(dim=0) 
    return x.t().matmul(x) / (x.size(0) - 1)

def corrcoef(x=None):
    c = covariance(x)
    std = c.diagonal(0).sqrt()
    c /= std[:,None] * std[None,:]
    return c