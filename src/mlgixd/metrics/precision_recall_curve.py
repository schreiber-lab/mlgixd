# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import numpy as np

from .full_metrics import FullMetrics


def get_precision_recall_curve(metrics: FullMetrics, interpolate_precision: bool = True):
    scores = np.concatenate([metrics.matched_scores, metrics.fp_scores])
    labels = np.concatenate([np.ones_like(metrics.matched_scores), np.zeros_like(metrics.fp_scores)])

    if not labels.size:
        return [], [], [], 0

    num_fn = len(metrics.false_negatives) + len(metrics.matched_ious)
    num_fp, matched = 0, 0
    indices = np.argsort(scores)[::-1]
    recalls, precisions, accuracies = [], [], []

    labels = labels[indices]

    for total, label in enumerate(labels):
        if label:
            matched += 1
            num_fn -= 1
        else:
            num_fp += 1

        recalls.append(matched / (matched + num_fn))
        precisions.append(matched / (matched + num_fp))
        accuracies.append(matched / (matched + num_fp + num_fn))

    if interpolate_precision:
        _interpolate_precisions(precisions)

    return recalls, precisions, accuracies


def get_max_accuracy_metrics(recalls, precisions, accuracies):
    idx = np.argmax(accuracies)
    return recalls[idx], precisions[idx], accuracies[idx]


def _interpolate_precisions(precisions: List[float]):
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
