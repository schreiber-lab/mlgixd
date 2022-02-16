# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    List,
    Tuple,
    Callable,
)

import torch
import torch.nn.functional as F
from torch import Tensor

from .encode_boxes import encode_boxes
from .utils import (
    assign_targets_to_anchors,
    Matcher,
    BalancedPositiveNegativeSampler,
)

__all__ = [
    'calc_losses',
    'calc_reg_box_loss',
    'calc_objectness_loss'
]


def calc_losses(
        matcher: Matcher,
        sampler: BalancedPositiveNegativeSampler,
        anchors: List[Tensor],
        targets: List[Tensor],
        objectness: Tensor,
        bbox_reg: Tensor,
        box_similarity: Callable[[Tensor, Tensor], Tensor]
) -> Tuple[Tensor, Tensor]:
    labels, matched_gt_boxes = assign_targets_to_anchors(matcher, anchors, targets, box_similarity)
    sampled_ind, sampled_pos_ind = sampler(labels)

    labels = torch.cat(labels, dim=0)
    regression_targets = encode_boxes(torch.cat(anchors), torch.cat(matched_gt_boxes))

    objectness_loss = calc_objectness_loss(labels[sampled_ind], objectness[sampled_ind])
    box_reg_loss = calc_reg_box_loss(
        regression_targets[sampled_pos_ind],
        bbox_reg[sampled_pos_ind],
        sampled_ind.numel())

    return objectness_loss, box_reg_loss


def calc_objectness_loss(labels, objectness):
    return F.binary_cross_entropy_with_logits(objectness.squeeze(), labels.squeeze())


def calc_reg_box_loss(regression_targets, pred_bbox_deltas, norm: int):
    return F.smooth_l1_loss(
        pred_bbox_deltas,
        regression_targets,
        beta=1 / 9,
        reduction='sum',
    ) / norm
