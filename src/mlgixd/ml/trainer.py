# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    Type,
    Tuple,
)
from time import perf_counter

import torch
from tqdm import tqdm
import numpy as np

from .model_mixin import ModelType
from .losses import Losses


class Trainer(object):
    def __init__(self, model: ModelType, dset, lr: float, optim: Type[torch.optim.Optimizer] = None):
        optim = optim or torch.optim.AdamW
        self.losses = Losses()
        self.dset = dset
        self.model = model
        self.optim = optim(model.parameters(), lr)
        self.callback_params = {}
        self.lrs = []
        self.training_duration = 0

    def state_dict(self, include_optim: bool = False):
        state_dict = {
            'losses': self.losses.state_dict(),
            'lrs': np.array(self.lrs),
            'model': self.model.cpu_state_dict(),
            'training_duration': self.training_duration,
        }

        if include_optim:
            state_dict['optim'] = self.optim.state_dict()

        return state_dict


    def train(self,
              num_batches: int = 2000,
              batch_size: int = 8,
              callbacks: Tuple['TrainerCallback', ...] or 'TrainerCallback' = (),
              disable_tqdm: bool = False
              ):

        start_time = perf_counter()

        if not isinstance(callbacks, TrainerCallback):
            callbacks = StackedTrainerCallbacks(callbacks)

        self.model.train()
        losses = []

        callbacks.start_training(self)

        with tqdm(total=num_batches, disable=disable_tqdm) as pbar:
            for i in range(num_batches):
                imgs, targets = self.dset.get_batch(batch_size)
                self.optim.zero_grad()
                loss_dict = self.model(imgs, targets)
                self.losses.add_dict(loss_dict)
                loss = sum(loss for loss in loss_dict.values())
                loss.backward()
                self.optim.step()
                loss = loss.item()
                pbar.update(1)
                pbar.set_description(f'Loss = {loss:.3e}')
                losses.append(loss)
                self.lrs.append(self.lr)

                callbacks.end_batch(self, i)

        callbacks.end_training(self)

        self.training_duration = perf_counter() - start_time

        return losses

    @property
    def lr(self):
        return self.optim.param_groups[0]['lr']

    @lr.setter
    def lr(self, lr: float):
        self.optim.param_groups[0]['lr'] = lr


class TrainerCallback(object):
    def start_training(self, trainer: Trainer) -> None:
        pass

    def end_training(self, trainer: Trainer) -> None:
        pass

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        pass


class StackedTrainerCallbacks(TrainerCallback):
    def __init__(self, callbacks: Tuple[TrainerCallback, ...]):
        self.callbacks = tuple(callbacks)

    def start_training(self, trainer: Trainer) -> None:
        for c in self.callbacks:
            c.start_training(trainer)

    def end_training(self, trainer: Trainer) -> None:
        for c in self.callbacks:
            c.end_training(trainer)

    def end_batch(self, trainer: Trainer, batch_num: int):
        for c in self.callbacks:
            c.end_batch(trainer, batch_num)
