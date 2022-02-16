# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
from collections import OrderedDict
from tqdm import trange

import torch
from torch import Tensor
from torch.nn import Module

from ..tools import (
    get_model_size,
    get_params_num,
    EvaluateFPS,
)
from ..simulations import SimDataset

ModelType = Union[Module, 'ModelMixin']


class ModelMixin:

    def get_params_num(self: ModelType) -> int:
        return get_params_num(self)

    def get_model_size(self: ModelType) -> str:
        return get_model_size(self)

    def cpu_state_dict(self: ModelType) -> 'OrderedDict[str, Tensor]':
        return _convert_state_dict(self.state_dict())

    def load_model(self: ModelType, name: str):
        state_dict = torch.load(name)
        if 'model' in state_dict:
            try:
                model = self.load_state_dict(state_dict['model'])
                return model
            except RuntimeError:
                pass
        return self.load_state_dict(state_dict)

    def save_model(self: ModelType, name: str):
        torch.save(self.state_dict(), name)

    @property
    def is_cuda(self: ModelType) -> bool:
        return next(self.parameters()).is_cuda

    @property
    def device(self: ModelType) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def default_model(cls, *args, **kwargs):
        raise NotImplementedError

    def evaluate_fps(self: ModelType, dset: SimDataset, num: int = 1000, *, disable_tqdm: bool = True) -> EvaluateFPS:
        evaluate_fps = EvaluateFPS()
        self.eval()
        for _ in trange(num, disable=disable_tqdm):
            imgs, _ = dset.get_batch(1)
            torch.cuda.synchronize()
            with evaluate_fps():
                _ = self(imgs)
                torch.cuda.synchronize()

        return evaluate_fps


def _convert_state_dict(state_dict, device: torch.device = torch.device('cpu')):
    return OrderedDict([(k, v.to(device)) for k, v in state_dict.items()])
