# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from torch import stack
from torch import Tensor


class BoxWidthPadding(object):
    def __init__(self, const_pad: float = 1.5, lin_pad: float = 0.1):
        self.const_pad = const_pad
        self.lin_pad = lin_pad

    def __call__(self, boxes_list: List[Tensor]) -> List[Tensor]:
        transformed_boxes: List[Tensor] = []

        for boxes in boxes_list:
            widths = boxes[:, 2] - boxes[:, 0]
            pad = widths * self.lin_pad + self.const_pad
            transformed_boxes.append(stack([boxes[:, 0] - pad, boxes[:, 1], boxes[:, 2] + pad, boxes[:, 3]], dim=1))

        return transformed_boxes
