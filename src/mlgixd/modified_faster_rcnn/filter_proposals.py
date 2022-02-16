# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List

import torch
from torch import nn, Tensor
from torchvision.ops import (
    clip_boxes_to_image,
    remove_small_boxes,
    batched_nms
)

from .utils import get_levels, get_top_n_idx


class FilterProposals(nn.Module):
    def __init__(self,
                 pre_nms_top_n_train=2000,
                 pre_nms_top_n_test=1000,
                 post_nms_top_n_train=2000,
                 post_nms_top_n_test=1000,
                 nms_thresh=0.7,
                 score_thresh=0.0,
                 min_size: float = 1e-3,
                 img_shape: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.img_shape = img_shape

        self._pre_nms_top_n_train = pre_nms_top_n_train
        self._pre_nms_top_n_test = pre_nms_top_n_test
        self._post_nms_top_n_train = post_nms_top_n_train
        self._post_nms_top_n_test = post_nms_top_n_test

        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = min_size

    @property
    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n_train
        return self._pre_nms_top_n_test

    @property
    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n_train
        return self._post_nms_top_n_test

    def filter_score_before_nms(self,
                                objectness: Tensor,
                                levels: Tensor,
                                proposals: Tensor,
                                num_anchors_per_level: List[int],
                                ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Filters proposals by choosing by choosing top k = self.pre_nms_top_n proposals with respect to
        objectness independently for each feature map and for each image.


        Args:
            objectness (Tensor): Objectness tensor;
                                 shape (num_images, sum(num_anchors_per_level)).
            levels (Tensor): Indices corresponding to source feature map;
                             shape = (num_images, sum(num_anchors_per_level)).
            proposals (Tensor): Proposal boxes;
                                shape = (num_images, sum(num_anchors_per_level), 4)
            num_anchors_per_level (List[int]): Number of anchors per feature map.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Filtered tensors (objectness, levels, proposals).
        """

        top_n_idx = get_top_n_idx(self.pre_nms_top_n, objectness, num_anchors_per_level)

        image_range = torch.arange(objectness.shape[0], device=objectness.device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        return objectness, levels, proposals

    def forward(self,
                proposals: Tensor,
                objectness: Tensor,
                num_anchors_per_level: List[int]
                ) -> Tuple[List[Tensor], List[Tensor]]:
        """

        Filters proposals by five stages:
            1. Filters by objectness score (leaves top k1 = self.pre_nms_top_n proposals
            independently for each feature map and for each image).
            2. Removes small boxes (width or height < self.min_size).
            3. Removes proposals with scores below self.score_thresh.
            4. Filters proposals via nms (iou <= self.nms_thresh) independently per feature map.
            5. Keeps top k2 = self.post_nms_top_n proposals per IMAGE.

        Thus, the maximum number of proposals per feature is self.post_nms_top_n.

        Total number of anchors per level A = sum(num_anchors_per_level).
        Number of images = N.

        Args:
            proposals (Tensor[N, A, 4]): Proposal boxes.
            objectness (Tensor[N, A]) Proposal scores.
            num_anchors_per_level (List[int]): Number of anchors per feature map.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: filtered_boxes, filtered_scores.
        """

        img_shape = self.img_shape
        num_images = proposals.shape[0]
        device = proposals.device

        objectness = objectness.view(num_images, -1)

        levels = get_levels(num_images, num_anchors_per_level, device)

        # select top_n boxes independently per level before applying nms
        objectness, levels, proposals = self.filter_score_before_nms(
            objectness, levels, proposals, num_anchors_per_level
        )

        objectness_prob = torch.sigmoid(objectness)

        filtered_boxes = []
        filtered_scores = []

        for boxes, scores, lvl in zip(proposals, objectness_prob, levels):
            boxes = clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only top k scoring predictions
            keep = keep[:self.post_nms_top_n]
            boxes, scores = boxes[keep], scores[keep]

            filtered_boxes.append(boxes)
            filtered_scores.append(scores)

        return filtered_boxes, filtered_scores
