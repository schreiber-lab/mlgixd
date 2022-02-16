# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

from torch import Tensor, nn
import torch.nn.functional as F

from .backbone import BackboneMixin


class FeaturePyramidNetwork(nn.Module):
    def __init__(
            self,
            in_channels_list: Tuple[int, ...],
            out_channels: int,
    ):
        super().__init__()

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """

        last_inner = self.inner_blocks[-1](features[-1])

        results = [
            self.layer_blocks[-1](last_inner)
        ]

        for idx in range(len(features) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](features[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        return results


class BackboneWithFPN(nn.Module, BackboneMixin):
    def __init__(self, backbone, backbone_channels: Tuple[int, ...], out_channels: int):
        super().__init__()
        self.backbone = backbone
        self._out_channels = out_channels
        self.fpn = FeaturePyramidNetwork(backbone_channels, out_channels)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, imgs: Tensor) -> List[Tensor]:
        features = self.backbone(imgs)
        features = self.fpn(features)
        return features
