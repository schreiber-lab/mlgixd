# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import torch
from torch import Tensor
from tqdm import trange, tqdm

import numpy as np

from ..ml import standard_filter, filter_score

from .match_criteria import MatchCriterion, get_match_function
from .scalar_metrics import ScalarMetrics
from .full_metrics import (
    FullMetrics,
    MatchedPair,
    FalseNegative,
    FalsePositive,
)


@torch.no_grad()
def get_sim_metrics(model, sim, num: int = 10000,
                    nms_level: float = 0.1,
                    score_level: float = 0.5,
                    match_threshold: float = 0.1,
                    match_criterion: Union[MatchCriterion, str] = MatchCriterion.iou,
                    disable_tqdm: bool = False,
                    scalar: bool = False,
                    **kwargs
                    ) -> FullMetrics or ScalarMetrics:
    model.eval()

    calc_metrics_func = get_scalar_metrics if scalar else get_full_metrics

    metrics = ScalarMetrics() if scalar else FullMetrics()

    for _ in trange(num, disable=disable_tqdm):
        img, boxes = sim.simulate_img()
        predictions, scores = model(img[None, None])
        predictions, scores = standard_filter(
            predictions[0], scores[0],
            nms_level=nms_level,
            score_level=score_level
        )
        metrics += calc_metrics_func(
            boxes, predictions, scores,
            score_threshold=0,
            match_threshold=match_threshold,
            match_criterion=match_criterion,
            **kwargs
        )

    return metrics


@torch.no_grad()
def get_batch_sim_metrics(
        targets: List[Tensor],
        predictions: List[Tensor],
        scores: List[Tensor],
        nms_level: float = 0.1,
        score_threshold: float = 0.5,
        match_threshold: float = 0.1,
        match_criterion: MatchCriterion = MatchCriterion.iou,
        disable_tqdm: bool = True,
        scalar: bool = True,
        **kwargs

) -> FullMetrics or ScalarMetrics:
    calc_metrics_func = get_scalar_metrics if scalar else get_full_metrics

    metrics = ScalarMetrics() if scalar else FullMetrics()

    for t, p, s in tqdm(zip(targets, predictions, scores), total=len(targets), disable=disable_tqdm):
        p, s = standard_filter(
            p, s,
            nms_level=nms_level,
            score_level=score_threshold
        )
        metrics += calc_metrics_func(
            t, p, s,
            score_threshold=0,
            match_threshold=match_threshold,
            match_criterion=match_criterion,
            **kwargs
        )

    return metrics


@torch.no_grad()
def get_scalar_metrics(
        target: Tensor,
        predicted: Tensor,
        scores: Tensor,
        score_threshold: float = 0.5,
        match_threshold: float = 0.1,
        match_criterion: Union[MatchCriterion, str] = MatchCriterion.iou,
        **kwargs
) -> ScalarMetrics:
    if isinstance(match_criterion, str):
        match_criterion = MatchCriterion(match_criterion)

    predicted, scores = filter_score(predicted, scores, score_threshold)

    func = get_match_function(match_criterion)

    metric_mtx, row_ind, col_ind = func(target, predicted, match_threshold, **kwargs)

    num_targets, num_predicted = target.shape[0], predicted.shape[0]

    matched_num = row_ind.size

    return ScalarMetrics(matched_num, num_predicted - matched_num, num_targets - matched_num)


@torch.no_grad()
def get_full_metrics(
        target: Tensor,
        predicted: Tensor,
        scores: Tensor,
        intensities: np.ndarray = None,
        score_threshold: float = 0.5,
        match_threshold: float = 1.,
        match_criterion: Union[MatchCriterion, str] = MatchCriterion.q_rel, **kwargs
):
    if intensities is None:
        intensities = - np.ones(target.shape[0])

    if isinstance(match_criterion, str):
        match_criterion = MatchCriterion(match_criterion)

    intensities = torch.from_numpy(intensities)

    predicted, scores = filter_score(predicted, scores, score_threshold)

    func = get_match_function(match_criterion)

    iou_mtx, row_ind, col_ind = func(target, predicted, match_threshold, **kwargs)

    missed_indices = _get_nonselected_indices(row_ind, target.shape[0])
    fp_indices = _get_nonselected_indices(col_ind, predicted.shape[0])

    matched_pairs = [
        MatchedPair(t_box, p_box, iou, score, intensity)
        for t_box, p_box, iou, score, intensity in zip(
            target[row_ind].tolist(),
            predicted[col_ind].tolist(),
            iou_mtx[row_ind, col_ind].tolist(),
            scores[col_ind].tolist(),
            intensities[row_ind].tolist()
        )
    ]

    false_positives = [
        FalsePositive(p_box, score) for p_box, score in zip(
            predicted[fp_indices].tolist(), scores[fp_indices].tolist()
        )
    ]

    false_negatives = [
        FalseNegative(t_box, intensity) for t_box, intensity in zip(
            target[missed_indices].tolist(), intensities[missed_indices].tolist()
        )
    ]

    return FullMetrics(
        matched_pairs, false_positives, false_negatives,
        [len(matched_pairs)], [len(false_positives)], [len(false_negatives)]
    )


def _get_nonselected_indices(selected_indices: np.ndarray, size: int):
    return np.array(list(set(np.arange(size)).difference(selected_indices)))
