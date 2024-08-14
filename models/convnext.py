# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in under
# https://github.com/keyu-tian/SparK/blob/main/LICENSE ur.
#
# This file is basically a copy of https://github.com/facebookresearch/ConvNeXt/blob/06f7b05f922e21914916406141f50f82b4a15852/models/convnext.py, with sparsity added.

from typing import List

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, SelectAdaptivePool2d
from timm.models.registry import register_model

from models.sparse_encoder import SparseConvNeXtBlock, SparseConvNeXtLayerNorm


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., global_pool='avg',
                 sparse=True,
                 ):
        super().__init__()
        self.dims: List[int] = dims
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            SparseConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first", sparse=sparse)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                SparseConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first", sparse=sparse),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[SparseConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                                      layer_scale_init_value=layer_scale_init_value, sparse=sparse) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.depths = depths

        self.num_features = dims[-1]
        
        self.apply(self._init_weights)
        if global_pool == '':
            self.global_pool = nn.Identity()
        else:
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        if num_classes > 0:
            self.norm = SparseConvNeXtLayerNorm(dims[-1], eps=1e-6, sparse=False)  # final norm layer for LE/FT; should not be sparse
            self.fc = nn.Linear(dims[-1], num_classes)
        else:
            self.norm = nn.Identity()
            self.fc = nn.Identity()
        
        self._sparse = sparse
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
    def get_downsample_ratio(self) -> int:
        return 32
    
    def get_feature_map_channels(self) -> List[int]:
        return self.dims
    
    def forward(self, x, hierarchical=False):
        ls = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            ls.append(x)
        if hierarchical:
            return ls
        else:
            return self.fc(self.norm(self.global_pool(x))) # (B, C, H, W) =mean=> (B, C) =norm&fc=> (B, NumCls)
    
    def get_classifier(self):
        return self.fc
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}, layer_scale_init_value={self.layer_scale_init_value:g}'

    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, value):
        self._sparse = value
        for name, m in self.named_modules():
            if name == 'norm':
                continue
            if isinstance(m, SparseConvNeXtLayerNorm):
                m.sparse = value
            if isinstance(m, SparseConvNeXtBlock):
                m.sparse = value


@register_model
def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


@register_model
def convnext_small(pretrained=False, in_22k=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model



if __name__ == '__main__':
    import timm
    import models.sparse_encoder as sparse_encoder

    cnn = timm.create_model('convnext_tiny')
    sparse_encoder._cur_active = torch.ones((1,1,1,1,))
    t = torch.ones((4,3,224,224))
    print(cnn(t).shape)
    
    cnn = timm.create_model('convnext_tiny', num_classes=0, global_pool='')
    sparse_encoder._cur_active = torch.ones((1,1,1,1,))
    t = torch.ones((4,3,224,224))
    print(cnn(t).shape)
    print([o.shape for o in cnn(t, hierarchical=True)])
    
    