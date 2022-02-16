# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
from torch.optim import lr_scheduler

from .trainer import (
    TrainerCallback,
    Trainer,
)


class SaveBestModel(TrainerCallback):
    def __init__(self,
                 loss_key: str,
                 model_path: Path = None,
                 load_best_model: bool = False,
                 init_loss: float = None,
                 save_every_epoch: bool = False,
                 ):
        self._state_dict = None
        self._loss_key = loss_key
        self._val_loss = init_loss
        self._best_epoch = 0
        self._load_best_model = load_best_model
        self.model_path = model_path or Path('models')
        self.save_every_epoch = save_every_epoch

    def clear(self):
        self._state_dict = None
        self._val_loss = None
        self._best_epoch = 0

    def save_model(self):
        if self.model_path and self._state_dict:
            torch.save(self._state_dict, self.model_path)

    def load_best_model(self, trainer):
        if self._state_dict:
            trainer.model.load_state_dict(self._state_dict)

    def start_training(self, trainer: Trainer):
        self.clear()
        if 'best_epoch' in trainer.callback_params:
            del trainer.callback_params['best_epoch']

    def end_training(self, trainer: Trainer):
        if self._load_best_model:
            self.load_best_model(trainer)

        if not self.save_every_epoch:
            self.save_model()

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        if self._val_loss is None or self._val_loss > trainer.losses[self._loss_key][-1]:
            self._val_loss = trainer.losses[self._loss_key][-1]
            self._state_dict = trainer.model.cpu_state_dict()
            self._best_epoch = batch_num
            trainer.callback_params['best_epoch'] = self._best_epoch

            if self.save_every_epoch:
                self.save_model()


class LRSchedulerCallback(TrainerCallback):
    def __init__(self, scheduler_class, *, freq: int = 1, fixed_after: int = 0, **kwargs):
        self._scheduler_class = scheduler_class
        self._freq = freq
        self._fixed_after = fixed_after
        self._kwargs = kwargs
        self.lr_list = []
        self.scheduler = None

    def start_training(self, trainer: Trainer) -> None:
        self.scheduler = self._scheduler_class(trainer.optim, **self._kwargs)

    def end_batch(self, trainer: Trainer, batch_num: int) -> None:
        if not (batch_num + 1) % self._freq and not (bool(self._fixed_after) and batch_num >= self._fixed_after):
            self.scheduler.step()
        self.lr_list.append(trainer.lr)

    def clear(self):
        self.lr_list.clear()


class ExpDecayLRCallback(LRSchedulerCallback):
    def __init__(self, gamma: float = 0.99, **kwargs):
        kwargs['gamma'] = gamma
        super().__init__(lr_scheduler.ExponentialLR, **kwargs)
