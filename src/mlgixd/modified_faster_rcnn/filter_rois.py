# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List

import torch
from torch import nn, Tensor

from torchvision.ops import (
    remove_small_boxes,
    batched_nms,
    clip_boxes_to_image
)

from .utils import get_levels


class FilterRois(nn.Module):
    def __init__(self,
                 nms_thresh=0.7,
                 score_thresh=0.05,
                 min_size: float = 1e-3,
                 max_num_per_image: int = 250,
                 img_shape: Tuple[int, int] = (512, 512)
                 ):
        super().__init__()
        self.img_shape = img_shape
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = min_size
        self.max_num_per_image = max_num_per_image

    def forward(self,
                proposals: Tensor,
                scores: Tensor,
                num_boxes_per_img: List[int]
                ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Filters final roi predictions by stages:
            1. Remove low scoring boxes (score < self.score_thresh).
            2. Remove small boxes (weight or height < self.min_size).
            3. Filter by nms (iou >= self.nms_thresh)
            4. Keep top self.max_num_per_image rois with respect to scores.

        Total number of proposals = P = sum(num_boxes_per_img)

        Args:
            proposals (Tensor[P, 4]): Detected boxes.
            scores (Tensor[P]): Confidence scores.
            num_boxes_per_img (List[int]): Number of boxes per image.

        Returns:
            Tuple[List[Tensor[N, 4]], List[Tensor[N]]]: filtered_boxes, filtered_scores
        """

        num_images: int = len(num_boxes_per_img)
        img_indices = get_levels(0, num_boxes_per_img, proposals.device)
        score_prob = torch.sigmoid(scores) * 2 - 1

        # Clip proposals to image shape
        proposals = clip_boxes_to_image(proposals, self.img_shape)

        # Remove low scoring boxes (score < self.score_thresh).
        keep = score_prob >= self.score_thresh
        img_indices, score_prob, proposals = img_indices[keep], score_prob[keep], proposals[keep]

        # Remove small boxes (weight or height < self.min_size).
        keep = remove_small_boxes(proposals, self.min_size)
        img_indices, score_prob, proposals = img_indices[keep], score_prob[keep], proposals[keep]

        # Filter by nms (iou >= self.nms_thresh).
        keep = batched_nms(proposals, score_prob, img_indices, self.nms_thresh)
        img_indices, score_prob, proposals = img_indices[keep], score_prob[keep], proposals[keep]

        # Separate by images.
        # Keep top self.max_num_per_image rois with respect to scores (already sorted by batched_nms)
        filtered_scores: List[Tensor] = [score_prob[img_indices == i][:self.max_num_per_image]
                                         for i in range(num_images)]
        filtered_boxes: List[Tensor] = [proposals[img_indices == i][:self.max_num_per_image]
                                        for i in range(num_images)]

        return filtered_boxes, filtered_scores
