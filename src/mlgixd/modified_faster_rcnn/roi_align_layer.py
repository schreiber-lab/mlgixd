# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor

from torchvision.ops.roi_align import RoIAlign as BasicRoiAlign


class ChooseFeatureMaps(nn.Module):
    def forward(self, boxes: Tensor) -> List[Tensor]:
        raise NotImplementedError


class ChooseByHeight(ChooseFeatureMaps):
    def __init__(self, height_thresh: float = 250):
        super().__init__()
        self.register_buffer('height_thresh', torch.tensor(height_thresh))

    def forward(self, boxes: Tensor) -> List[Tensor]:
        heights = boxes[:, 3] - boxes[:, 1]
        first_map_indices: Tensor = heights < self.height_thresh
        second_map_indices: Tensor = ~first_map_indices

        return [first_map_indices, second_map_indices]


class ChooseFirstMap(ChooseFeatureMaps):
    def forward(self, boxes: Tensor) -> List[Tensor]:
        return [torch.ones(*boxes.shape[:-1], dtype=torch.bool, device=boxes.device)]


class ChooseOneMap(ChooseFeatureMaps):
    def __init__(self, idx: int = 0, num_of_maps: int = 1):
        super().__init__()
        self.idx = idx
        self.num_of_maps = num_of_maps

    def forward(self, boxes: Tensor) -> List[Tensor]:
        return [
            torch.ones(*boxes.shape[:-1], dtype=torch.bool, device=boxes.device)
            if i == self.idx else
            torch.zeros(*boxes.shape[:-1], dtype=torch.bool, device=boxes.device)
            for i in range(self.num_of_maps)
        ]


class RoiAlign(nn.Module):
    def __init__(self, height: int = 7, width: int = 7, choose_map: ChooseFeatureMaps = None,
                 feature_map_sizes: Tuple[Tuple[int, int], ...] = ((32, 128), (16, 128)),
                 img_size: Tuple[int, int] = (512, 512)
                 ):
        super().__init__()

        scales = _init_scales(feature_map_sizes, img_size)
        self.register_buffer('scales', torch.tensor(scales))
        self.choose_map = choose_map or ChooseByHeight()
        self.roi_align = BasicRoiAlign((height, width), 1., -1, aligned=False)

        self.size = height * width

    def forward(self, feature_maps: List[Tensor], boxes: List[Tensor]):
        boxes, img_indices = _cat_boxes(boxes)

        feature_indices = self.choose_map(boxes)

        assert len(feature_indices) == len(feature_maps)

        rescaled_boxes = []

        for feature_map, indices, scales in zip(feature_maps, feature_indices, self.scales):
            if torch.any(indices):
                scaled_boxes = _rescale_boxes(boxes[indices], scales)
                rois = torch.cat([img_indices[indices][:, None], scaled_boxes], dim=1)
                rescaled_boxes.append(self.roi_align(feature_map, rois))

        rescaled_boxes = torch.cat(rescaled_boxes)

        if len(feature_maps) > 1:
            indices_order = torch.argsort(torch.cat([torch.where(fi)[0] for fi in feature_indices]))
            rescaled_boxes = rescaled_boxes[indices_order]

        return rescaled_boxes


def _init_scales(feature_map_sizes: Tuple[Tuple[int, int], ...], img_size: Tuple[int, int]):
    return [(f[0] / img_size[0], f[1] / img_size[1]) for f in feature_map_sizes]


def _cat_boxes(boxes: List[Tensor]):
    device = boxes[0].device
    indices = torch.cat([
        torch.full((b.shape[0],), n, dtype=torch.int, device=device) for n, b in enumerate(boxes)
    ])
    boxes = torch.cat(boxes, 0)
    return boxes, indices


def _rescale_boxes(boxes: Tensor, scales: Tuple[float, float]):
    ys, xs = scales
    x0, y0, x1, y1 = boxes.split(1, dim=1)
    return torch.stack([x0 * xs, y0 * ys, x1 * xs, y1 * ys], dim=1)[:, :, 0]
