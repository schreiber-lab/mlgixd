# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

class ScalarMetrics(object):
    __slots__ = ('_matched_num', '_fp_num', '_fn_num')

    def __init__(self, matched_num: int = 0, fp_num: int = 0, fn_num: int = 0):
        self._matched_num = matched_num
        self._fp_num = fp_num
        self._fn_num = fn_num

    def clear(self):
        self._matched_num = 0
        self._fp_num = 0
        self._fn_num = 0

    @property
    def matched_num(self):
        return self._matched_num

    @property
    def fp_num(self):
        return self._fp_num

    @property
    def fn_num(self):
        return self._fn_num

    @property
    def recall(self):
        try:
            return self._matched_num / (self._matched_num + self._fn_num)
        except ZeroDivisionError:
            return 0

    @property
    def precision(self):
        try:
            return self._matched_num / (self._matched_num + self._fp_num)
        except ZeroDivisionError:
            return 0

    @property
    def accuracy(self):
        try:
            return self._matched_num / (self._matched_num + self._fn_num + self._fp_num)
        except ZeroDivisionError:
            return 0

    def __add__(self, other: 'ScalarMetrics'):
        if not isinstance(other, ScalarMetrics):
            return NotImplemented

        return ScalarMetrics(
            self.matched_num + other.matched_num,
            self.fp_num + other.fp_num,
            self.fn_num + other.fn_num,
        )

    def __iadd__(self, other: 'ScalarMetrics'):
        if not isinstance(other, ScalarMetrics):
            return NotImplemented

        self.append(other)

        return self

    def append(self, other: 'ScalarMetrics'):
        self._matched_num += other.matched_num
        self._fp_num += other.fp_num
        self._fn_num += other.fn_num

    def __repr__(self):
        return f'ScalarMetrics(matched_num={self.matched_num}, fp_num={self.fp_num}, fn_num={self.fn_num})'

    def print(self):
        print(self.metrics_str)

    @property
    def metrics_str(self) -> str:
        return f'Accuracy = {self.accuracy * 100:.2f} %' \
               f'; recall = {self.recall * 100:.2f} %' \
               f'; precision = {self.precision * 100:.2f} %'
