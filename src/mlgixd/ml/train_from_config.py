# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from ..config import Config
from .trainer import Trainer
from .trainer_callbacks import LRSchedulerCallback


def train_from_config(model, dset, config: Config) -> Trainer:
    train_config = config.training
    trainer = Trainer(model, dset, train_config.init_lr)
    callbacks = []

    if isinstance(train_config.lr_scheduler, type(LRSchedulerCallback)):
        callbacks.append(train_config.lr_scheduler(**train_config.lr_scheduler_kwargs))

    trainer.train(
        train_config.num_batches,
        train_config.batch_size,
        tuple(callbacks),
        disable_tqdm=config.general.disable_tqdm
    )
    return trainer
