# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import numpy as np
import torch


class Losses(defaultdict):
    def __init__(self):
        super().__init__(list)

    def state_dict(self):
        return {k: np.array(v) for k, v in self.items()}

    def add_dict(self, epoch_loss: dict):
        for k, v in epoch_loss.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self[k].append(v)

    def add_batch_dict(self, batch_loss: dict):
        for k, v in batch_loss.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self[f'batch_{k}'].append(v)

    def on_epoch_end(self):
        to_delete = []
        to_append = []

        for k, v in self.items():
            if k.startswith('batch_'):
                to_append.append((k[6:], np.mean(self[k])))  # len('batch_') == 6
                to_delete.append(k)

        for k in to_delete:
            del self[k]

        for k, v in to_append:
            self[k].append(v)
