# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
[CODE FROM] https://github.com/facebookresearch/ConvNeXt/blob/b2d5a2de8b04533cc288400584db756c29066109/models/convnext.py
modified by: STomoya (https://github.com/STomoya)
'''

from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Modified implementation:
    (3) DwConv -> GroupNorm (group_size=1)   -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (GroupNorm with group_size=1 is equivalent only to channel_last LayerNorm with channels=feat.size(1) when feats.ndim==4)

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Modifications:
        - no head
        - uses ModuleDict
        - channel_first LayerNorm -> GroupNorm group_size=1

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self,
            in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
            layer_scale_init_value=1e-6
        ):
        super().__init__()

        layers = []
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0], eps=1e-6)
        )
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            if i == 0: downsample_layer = stem
            else:
                downsample_layer = nn.Sequential(
                    nn.GroupNorm(1, dims[i-1], eps=1e-6),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2),
                )
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            cur += depths[i]
            layers.append((f'layer{i+1}', nn.Sequential(OrderedDict([
                ('down', downsample_layer), ('stage', stage)
            ]))))
        layers.extend([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten(1)),
            ('norm',    nn.LayerNorm(dims[-1], eps=1e-6))])
        self._feature_blocks = nn.ModuleDict(layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for _, stage in self._feature_blocks.items():
            x = stage(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x
