# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

from torch import Tensor
from torchvision.ops import box_iou

import numpy as np

from scipy.optimize import linear_sum_assignment


class MatchCriterion(Enum):
    iou = 'iou'
    q = 'q'
    q_rel = 'q_rel'


def get_match_function(
        match_criterion: MatchCriterion
):
    if match_criterion == MatchCriterion.iou:
        return get_iou_match
    if match_criterion == MatchCriterion.q:
        return get_q_match
    if match_criterion == MatchCriterion.q_rel:
        return get_q_rel_match
    raise ValueError(f'Unknown match criterion {match_criterion}')


def get_match(
        match_criterion: MatchCriterion,
        target: Tensor, predicted: Tensor,
        thresh: float = 0,
        **kwargs
):
    return get_match_function(match_criterion)(target, predicted, thresh, **kwargs)


def get_iou_match(
        target: Tensor, predicted: Tensor, thresh: float = 0, **kwargs
):
    iou_mtx = box_iou(target, predicted).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-iou_mtx)

    indices = iou_mtx[row_ind, col_ind] > thresh
    row_ind, col_ind = row_ind[indices], col_ind[indices]

    return iou_mtx, row_ind, col_ind


def get_q_match(
        target: Tensor, predicted: Tensor,
        thresh: float = 10., rel: bool = False, **kwargs
):
    q_mtx = calc_box_dq_mtx(target, predicted, rel=rel, **kwargs)
    row_ind, col_ind = linear_sum_assignment(q_mtx)

    indices = q_mtx[row_ind, col_ind] < thresh
    row_ind, col_ind = row_ind[indices], col_ind[indices]

    return q_mtx, row_ind, col_ind


def get_q_rel_match(
        target: Tensor, predicted: Tensor,
        thresh: float = 10, **kwargs
):
    return get_q_match(target, predicted, thresh, rel=True, **kwargs)


def calc_box_dq_mtx(target, predicted, rel: bool = False, min_iou: float = 0.5, **kwargs):
    iou_mtx = box_iou(target, predicted).cpu().numpy()
    qt = (target[:, 0] + target[:, 2]) / 2
    qp = (predicted[:, 0] + predicted[:, 2]) / 2
    q_mtx = np.abs(qt[:, None].cpu().numpy() - qp[None].cpu().numpy())

    if rel:
        widths = (target[:, 2] - target[:, 0]).cpu().numpy()
        q_mtx = q_mtx / widths[:, None]

    q_mtx[iou_mtx <= min_iou] = 1e7  # inf not supported by linear_sum_assignment

    return q_mtx
