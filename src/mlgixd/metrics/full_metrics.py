# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from typing import List

import numpy as np

from .scalar_metrics import ScalarMetrics
from ..tools import to_t, to_np

MatchedPair = namedtuple('MatchedPair', 't_box p_box iou score intensity')
FalsePositive = namedtuple('FalsePositive', 'p_box score')
FalseNegative = namedtuple('FalseNegative', 't_box intensity')


class FullMetrics(object):
    # MatchedPair: t_box p_box iou score intensity

    MATCHED_PAIR_KEYS = (
        'matched_t_boxes',
        'matched_p_boxes',
        'matched_ious',
        'matched_scores',
        'matched_intensities',
    )

    # FalsePositive: p_box score

    FP_KEYS = (
        'fp_boxes',
        'fp_scores',
    )

    # FalseNegative: t_box intensity

    FN_KEYS = (
        'fn_boxes',
        'missed_intensities',
    )

    NUM_KEYS = (
        'num_matched_per_image',
        'num_fp_per_image',
        'num_fn_per_image',
    )

    KEYS = (*MATCHED_PAIR_KEYS, *FP_KEYS, *FN_KEYS, *NUM_KEYS)

    def __init__(self,
                 matched_pairs: List[MatchedPair] = (),
                 false_positives: List[FalsePositive] = (),
                 false_negatives: List[FalseNegative] = (),
                 num_matched: List[int] = (),
                 num_fp: List[int] = (),
                 num_fn: List[int] = (),
                 ):

        self._matched_pairs = list(matched_pairs)
        self._fp = list(false_positives)
        self._fn = list(false_negatives)
        self._num_matched = list(num_matched)
        self._num_fp = list(num_fp)
        self._num_fn = list(num_fn)

        if not (
                len(self._num_matched) == len(self._num_fp) and
                len(self._num_matched) == len(self._num_fn)
        ):
            raise ValueError(
                'Inconsistent num lengths: ',
                len(self._num_matched),
                len(self._num_fp),
                len(self._num_fn),
            )

        if sum(self._num_matched) != len(self._matched_pairs):
            raise ValueError('Inconsistent matched num')
        if sum(self._num_fp) != len(self._fp):
            raise ValueError('Inconsistent fp num')
        if sum(self._num_fn) != len(self._fn):
            raise ValueError('Inconsistent fn num')

    def state_dict(self):
        return {k: to_t(getattr(self, k), device='cpu') for k in self.KEYS}

    @classmethod
    def load_state_dict(cls, state_dict):
        data_dict = {k: to_np(v) for k, v in state_dict.items()}
        return cls.from_dict(data_dict)

    @classmethod
    def from_dict(cls, data_dict):
        matched_pairs = [MatchedPair(*d) for d in zip(*[data_dict[key] for key in cls.MATCHED_PAIR_KEYS])]
        false_positives = [FalsePositive(*d) for d in zip(*[data_dict[key] for key in cls.FP_KEYS])]
        false_negatives = [FalseNegative(*d) for d in zip(*[data_dict[key] for key in cls.FN_KEYS])]

        return cls(
            matched_pairs=matched_pairs,
            false_positives=false_positives,
            false_negatives=false_negatives,
            num_matched=list(data_dict['num_matched_per_image']),
            num_fp=list(data_dict['num_fp_per_image']),
            num_fn=list(data_dict['num_fn_per_image']),
        )

    def get_q_error(self, min_score: float = 0) -> np.ndarray:
        indices = self.matched_scores > min_score if min_score else ...
        q_err = (
                        self.matched_p_boxes[indices, 0] + self.matched_p_boxes[indices, 2] -
                        self.matched_t_boxes[indices, 0] - self.matched_t_boxes[indices, 2]
                ) / 2
        return q_err

    @property
    def matched_ious(self) -> np.ndarray:
        return np.array([pair.iou for pair in self._matched_pairs])

    @property
    def num_matched_per_image(self) -> np.ndarray:
        return np.array(self._num_matched)

    @property
    def num_fp_per_image(self) -> np.ndarray:
        return np.array(self._num_fp)

    @property
    def num_fn_per_image(self) -> np.ndarray:
        return np.array(self._num_fn)

    @property
    def scalar_metrics(self) -> ScalarMetrics:
        return ScalarMetrics(len(self._matched_pairs), len(self._fp), len(self._fn))

    @property
    def matched_t_boxes(self) -> np.ndarray:
        return np.array([pair.t_box for pair in self._matched_pairs])

    @property
    def matched_p_boxes(self) -> np.ndarray:
        return np.array([pair.p_box for pair in self._matched_pairs])

    @property
    def fn_boxes(self) -> np.ndarray:
        return np.array([pair.t_box for pair in self._fn])

    @property
    def fp_boxes(self) -> np.ndarray:
        return np.array([pair.p_box for pair in self._fp])

    @property
    def matched_scores(self) -> np.ndarray:
        return np.array([pair.score for pair in self._matched_pairs])

    @property
    def fp_scores(self) -> np.ndarray:
        return np.array([fp.score for fp in self._fp])

    @property
    def matched_intensities(self) -> np.ndarray:
        return np.array([pair.intensity for pair in self._matched_pairs])

    @property
    def missed_intensities(self) -> np.ndarray:
        return np.array([fn.intensity for fn in self._fn])

    @property
    def matched_pairs(self):
        return list(self._matched_pairs)

    @property
    def false_positives(self):
        return list(self._fp)

    @property
    def false_negatives(self):
        return list(self._fn)

    def append(self, other: 'FullMetrics'):
        self._matched_pairs += other._matched_pairs
        self._fp += other._fp
        self._fn += other._fn
        self._num_matched += other._num_matched
        self._num_fp += other._num_fp
        self._num_fn += other._num_fn

    def __add__(self, other):
        if not isinstance(other, FullMetrics):
            return NotImplemented

        return FullMetrics(
            self._matched_pairs + other._matched_pairs,
            self._fp + other._fp,
            self._fn + other._fn,
            self._num_matched + other._num_matched,
            self._num_fp + other._num_fp,
            self._num_fn + other._num_fn,
        )

    def __iadd__(self, other):
        if not isinstance(other, FullMetrics):
            return NotImplemented

        self.append(other)
        return self

    def __eq__(self, other):
        if not isinstance(other, FullMetrics):
            return False

        for key in self.KEYS:
            if not np.allclose(getattr(self, key), getattr(other, key)):
                return False
        return True
