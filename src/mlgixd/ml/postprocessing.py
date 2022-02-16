# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
from torch import Tensor
from torchvision.ops import nms, batched_nms

from ..modified_faster_rcnn.utils import num_per_level_from_levels


def batch_filter(
        predictions: List[Tensor], scores: List[Tensor],
        nms_level: float = 0.1,
        score_level: float = 0.1
) -> Tuple[List[Tensor], List[Tensor]]:

    num_imgs = len(scores)
    img_indices = torch.cat([torch.full_like(s, i) for i, s in enumerate(scores)])
    predictions, scores = torch.cat(predictions), torch.cat(scores)
    keep = scores > score_level
    predictions, scores, img_indices = predictions[keep], scores[keep], img_indices[keep]
    keep = batched_nms(predictions, scores, img_indices, nms_level)
    keep = torch.sort(keep).values
    predictions, scores, img_indices = predictions[keep], scores[keep], img_indices[keep]
    img_nums = num_per_level_from_levels(img_indices, num_imgs).tolist()
    predictions, scores = predictions.split(img_nums), scores.split(img_nums)

    return predictions, scores


def filter_nms(predictions: Tensor, scores: Tensor, level: float = 0.1):
    if not scores.numel():
        return predictions, scores

    indices = nms(predictions, scores, level)
    return predictions[indices], scores[indices]


def filter_score(predictions: Tensor, scores: Tensor, level: float = 0.1):
    if not level:
        return predictions, scores

    if not scores.numel():
        return predictions, scores

    indices = scores > level
    return predictions[indices], scores[indices]


def standard_filter(predictions: Tensor, scores: Tensor,
                    nms_level: float = 0.1,
                    score_level: float = 0.8,
                    ):
    predictions, scores = filter_score(predictions, scores, score_level)
    predictions, scores = filter_nms(predictions, scores, nms_level)

    return predictions, scores
