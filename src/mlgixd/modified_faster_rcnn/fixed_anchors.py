# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List

import torch
from torch import Tensor


class FixedAnchorsGenerator(object):
    def __init__(self,
                 height_weight_per_feature: Tuple[Tuple[Tuple[float, float], ...], ...],
                 img_shape: tuple = (512, 512),
                 feature_map_sizes: Tuple[int, ...] or Tuple[Tuple[int, int], ...] = ((32, 128), (16, 128)),
                 ):

        assert len(height_weight_per_feature) == len(feature_map_sizes)
        num_anchors_per_location = [len(pairs) for pairs in height_weight_per_feature]
        assert len(set(num_anchors_per_location)) == 1, 'Different anchors number per location is not supported'

        self.height_weight_per_feature = height_weight_per_feature
        self.img_shape = img_shape
        self.feature_map_sizes = _convert_feature_map_sizes(feature_map_sizes)
        self.anchors = self._generate_anchors()

    def set_img_shape(self, img_shape: Tuple[int, int], feature_map_sizes: Tuple[Tuple[int, int], ...]):
        self.img_shape = img_shape
        self.feature_map_sizes = _convert_feature_map_sizes(feature_map_sizes)
        self.update_anchors()

    def __call__(self, img_num: int, device: torch.device):
        return self.get_anchors(img_num, device)

    def num_anchors_per_location(self) -> int:
        return len(self.height_weight_per_feature[0])

    def update_anchors(self):
        self.anchors = self._generate_anchors()

    def _generate_anchors(self):
        device, dtype = torch.device('cpu'), torch.float32

        anchors = []

        strides_width = [self.img_shape[1] // s[1] for s in self.feature_map_sizes]
        strides_height = [self.img_shape[0] // s[0] for s in self.feature_map_sizes]

        for stride_width, stride_height, f_size, height_weight_pairs in zip(
                strides_width,
                strides_height,
                self.feature_map_sizes,
                self.height_weight_per_feature
        ):
            base_anchors = _generate_base_anchors(height_weight_pairs, device=device, dtype=dtype)

            shifts = _generate_anchor_shifts(stride_width, stride_height, f_size, dtype, device)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return torch.cat(anchors)

    def get_anchors(self, img_num: int, device: torch.device) -> List[Tensor]:
        if device != self.anchors.device:
            self.anchors = self.anchors.to(device)
        return [self.anchors for _ in range(img_num)]


def _convert_feature_map_sizes(feature_map_sizes):
    return [(s, s) if isinstance(s, int) else s for s in feature_map_sizes]


def _generate_anchor_shifts(stride_width, stride_height, f_size, dtype, device):
    shifts_x = torch.arange(
        0, f_size[1], dtype=dtype, device=device
    ) * stride_width

    shifts_y = torch.arange(
        0, f_size[0], dtype=dtype, device=device
    ) * stride_height

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)

    return torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)


def _generate_base_anchors(height_weight_pairs: Tuple[Tuple[float, float], ...],
                           device: torch.device = 'cuda',
                           dtype=torch.float32):
    sizes = torch.as_tensor(height_weight_pairs, dtype=dtype, device=device)
    hs, ws = sizes[:, 0], sizes[:, 1]
    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
    return base_anchors
