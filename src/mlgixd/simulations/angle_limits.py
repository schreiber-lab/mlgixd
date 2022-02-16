# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import random
from math import pi, sin, cos

import torch
from torch import Tensor


class AngleLimits(object):
    def __init__(self,
                 slope_range: tuple = (0, 0.1),
                 size_ratio_range: tuple = (-0.05, 0.05),
                 r_size: int = 512,
                 phi_size: int = 512
                 ):
        self.r_size = r_size
        self.phi_size = phi_size
        self.slope_range = slope_range
        self.size_ratio_range = size_ratio_range
        self.slope, self.size_ratio = None, None
        self._x_size, self._y_size = None, None
        self.update_params()

    def max(self, r: Tensor) -> Tensor:
        return (
                       (r <= self._y_size) +
                       (r > self._y_size) *
                       torch.nan_to_num(torch.arcsin(self._y_size / r)) / pi * 2
               ) * self.phi_size

    def min(self, r: Tensor) -> Tensor:
        dark_area = r * self.slope
        geometry_area = (r > self._x_size) * torch.nan_to_num(torch.arccos(self._x_size / r)) / pi * 2 * self.phi_size
        return torch.maximum(geometry_area, dark_area)

    def update_params(self):
        self.slope = random.uniform(*self.slope_range)
        self.size_ratio = random.uniform(*self.size_ratio_range)
        x_weight = pi / 4 - self.size_ratio

        self._x_size = self.r_size * sin(x_weight)
        self._y_size = self.phi_size * cos(x_weight)
