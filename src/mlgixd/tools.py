# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from time import perf_counter
from contextlib import contextmanager

import torch
from torch import Tensor
from torch.nn import Module


__all__ = [
    'EvaluateTime',
    'EvaluateFPS',
    'to_np',
    'to_t',
    'tensor_size',
    'get_size_str',
    'get_params_num',
    'get_model_size',
]


class EvaluateTime(list):
    @contextmanager
    def __call__(self):
        start = perf_counter()
        yield
        self.append(perf_counter() - start)
        
    def __repr__(self):
        return f'EvaluateTime(total={sum(self)}, num_records={len(self)})'


class EvaluateFPS(EvaluateTime):
    @property
    def fps(self):
        return len(self) / sum(self)

    def print(self):
        print(f'Average FPS = {self.fps:.1f}')
        print('Number of records =', len(self))

    def __repr__(self):
        return f'EvaluateFPS(fps={self.fps:.1f}, num_frames={len(self)})'


def to_np(arr):
    if isinstance(arr, Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


def to_t(t, *, dtype=torch.float, device='cuda'):
    if isinstance(t, Tensor):
        return t
    return torch.tensor(t, dtype=dtype, device=device)


def tensor_size(t):
    size = t.element_size() * t.nelement()
    print(get_size_str(size))


def get_size_str(n_bytes):
    if n_bytes >= 1000 ** 3:
        size_str = f'{(n_bytes / 1000 ** 3):.2f} Gb'
    elif n_bytes >= 1000 ** 2:
        size_str = f'{(n_bytes / 1000 ** 2):.2f} Mb'
    else:
        size_str = f'{(n_bytes / 1000):.2f} Kb'
    return size_str


def get_params_num(model: Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_size(model: Module) -> str:
    return get_size_str(get_params_num(model) * 4)  # 32-bit floating number == 4 bytes
