# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List

import torch
from torch.nn.functional import relu
from torch import nn
from torch import Tensor

import numpy as np


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()

        self.in_channels = in_channels
        self.num_anchors = num_anchors

        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=(1, 1), stride=(1, 1))
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=(1, 1), stride=(1, 1)
        )

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, features: List[Tensor]) -> Tuple[Tensor, Tensor, List[int]]:
        objectness = []  # object / not object
        bbox_reg = []

        for feature in features:
            t = relu(self.conv(feature))
            objectness.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))

        num_anchors_per_level = [int(np.prod(o.shape[1:])) for o in objectness]
        objectness, bbox_reg = self._concat_output(objectness, bbox_reg)
        return objectness, bbox_reg, num_anchors_per_level

    @staticmethod
    def _concat_output(logits: List[Tensor], bbox_reg: List[Tensor]) -> Tuple[Tensor, Tensor]:
        box_cls_flattened = []
        box_regression_flattened = []

        for box_cls_per_level, box_regression_per_level in zip(
                logits, bbox_reg
        ):
            n, a, h, w = box_cls_per_level.shape

            box_cls_per_level = permute_and_flatten(box_cls_per_level, n, 1, h, w)
            box_regression_per_level = permute_and_flatten(box_regression_per_level, n, 4, h, w)

            box_cls_flattened.append(box_cls_per_level)
            box_regression_flattened.append(box_regression_per_level)

        box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
        box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
        return box_cls, box_regression


def permute_and_flatten(layer: Tensor, n: int, c: int, h: int, w: int) -> Tensor:
    layer = layer.view(n, -1, c, h, w)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(n, -1, c)
    return layer
