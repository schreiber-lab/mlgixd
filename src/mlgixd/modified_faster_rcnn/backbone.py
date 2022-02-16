# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union, List
from functools import lru_cache

import torch
from torch import nn
from torch import Tensor

from torchvision.models.resnet import (
    BasicBlock,
    conv1x1,
    resnet18
)

import numpy as np

from ..ml import ModelMixin, ModelType, init_kaiming


class BackboneMixin(ModelMixin):
    @property
    def out_channels(self):
        raise NotImplementedError

    @lru_cache()
    def feature_map_sizes(self, img_shape: Tuple[int, int] = (512, 512)) -> Tuple[Tuple[int, int], ...]:
        return _calc_feature_map_sizes(self, img_shape)


BackboneType = Union[nn.Module, BackboneMixin]  # TODO: Should be a generic type.


class VCompressResNet(nn.Module, BackboneMixin):
    @property
    def out_channels(self):
        return self._out_channels

    def __init__(
            self,
            layers: Tuple[int] = (2, 2, 2, 2),
            zero_init_residual: bool = True,
            groups: int = 1,
            width_per_group: int = 64,
            norm_layer=None,
            channels: Tuple[int, int, int, int] = (64, 128, 256, 256),
            init_from_resnet: bool = True,
            include_features_list: Tuple[int, ...] = (2, 3, 4),
            asymmetric: bool = True,
    ) -> None:

        super().__init__()

        self.include_features_list = tuple(include_features_list)

        self._out_channels = channels[-1]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        strides = (2, 1) if asymmetric else (2, 2)

        self.layer1 = self._make_layer(channels[0], layers[0])
        self.layer2 = self._make_layer(channels[1], layers[1], stride=strides)
        self.layer3 = self._make_layer(channels[2], layers[2], stride=strides)
        self.layer4 = self._make_layer(channels[3], layers[3], stride=strides)

        self._init_layers(zero_init_residual)

        if init_from_resnet:
            self.init_from_resnet()

    @torch.no_grad()
    def init_from_resnet(self):
        resnet = resnet18(True, False)

        for (b_name, b_m), (r_name, r_m) in zip(self.named_modules(), resnet.named_modules()):
            if isinstance(b_m, nn.Conv2d):
                assert b_name == r_name, f'{b_name} != {r_name}'

                b_weight = b_m.weight
                r_weight = r_m.weight

                if b_weight.shape == r_weight.shape:
                    b_weight.copy_(r_weight)

                else:
                    assert b_weight.shape[-2:] == r_weight.shape[-2:]

                    b_channels, r_channels = b_weight.shape[:2], r_weight.shape[:2]
                    b_size, r_size = np.prod(b_channels), np.prod(r_channels)
                    ratio = r_size // b_size

                    assert ratio > 0

                    indices = torch.randperm(int(b_size)) * ratio

                    sub_weights = r_weight.view(-1, *b_weight.shape[-2:])[indices].view(b_weight.shape)
                    b_weight.copy_(sub_weights)

    def _init_layers(self, zero_init_residual):
        init_kaiming(self)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    planes: int,
                    blocks: int,
                    stride: Union[int, Tuple[int, int]] = 1) -> nn.Sequential:
        block = BasicBlock
        norm_layer = self._norm_layer
        downsample = None

        if stride not in ((1, 1), 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, norm_layer=norm_layer)]

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> List[Tensor]:
        features: List[Tensor] = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate((
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
        ), 1):
            x = layer(x)
            if i in self.include_features_list:
                features.append(x)

        return features

    def forward(self, x: Tensor) -> List[Tensor]:
        return self._forward_impl(x)


@torch.no_grad()
def _calc_feature_map_sizes(model: ModelType, img_shape: Tuple[int, int]):
    img = torch.ones(1, 1, *img_shape, device=model.device)
    return tuple(tuple(f.shape[-2:]) for f in model(img))
