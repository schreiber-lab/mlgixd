# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from argparse import ArgumentParser
from pathlib import Path
import warnings

from .simulations import SimDataset
from .ml import train_from_config, init_model_from_config
from .metrics import get_sim_metrics_from_config
from .config import Config

__all__ = [
    'run',
    'run_from_config',
]


def run():
    warnings.filterwarnings("ignore", category=UserWarning)

    args_dict = _parse_config_args()

    if args_dict['generate']:
        Config.save_default(args_dict['config_path'])
        return

    config = Config.load(args_dict['config_path'])

    run_from_config(config)


def run_from_config(config: Config):
    logging.basicConfig(level=logging.INFO, format=f'{config.general.name}: %(asctime)s %(message)s')
    logger = logging.getLogger(__name__)

    state_dict = {'config': config.asdict()}

    model = init_model_from_config(config)

    logger.info(f'Model is loaded: {model.__class__.__name__}')

    if config.general.save_model_size:
        state_dict['num_parameters'] = model.get_params_num()
        state_dict['model_size'] = model.get_model_size()

    dset = SimDataset()

    if config.general.train:
        logger.info('Start training')

        trainer = train_from_config(model, dset, config)

        logger.info('Training is completed.')
        state_dict.update(trainer.state_dict())

    if config.general.evaluate_fps:
        logging.info('Evaluate FPS')

        evaluate_fps = model.evaluate_fps(dset, config.general.evaluate_fps)

        state_dict['evaluate_fps'] = {
            'records': list(evaluate_fps),
            'fps': evaluate_fps.fps
        }

    if config.general.test:
        logger.info('Start calculating metrics')

        state_dict.update(get_sim_metrics_from_config(model, dset.sim, config))

        logger.info('Metrics are calculated.')

    if config.general.save:

        dest_path = Path(config.general.dest_path)

        if not dest_path.parent.is_dir():
            dest_path.parent.mkdir()

        torch.save(state_dict, str(dest_path))

    logger.info(f'Successfully saved to {config.general.dest_path}')


def _parse_config_args() -> dict:
    parser = ArgumentParser(description='Train, test & save results for GIXD detectron model.')
    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--generate', '-G', action='store_true', help='Save the default config to the provided path')
    args = parser.parse_args()
    args_dict = {
        'config_path': Path(args.config),
        'generate': args.generate,
    }

    return args_dict
