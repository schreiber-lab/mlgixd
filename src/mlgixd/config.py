# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, List
import yaml
from copy import deepcopy
from dataclasses import dataclass, asdict, field, Field
from functools import lru_cache

from .package_info import __version__

__all__ = [
    'Config',
    'TrainerConfig',
    'ModelConfig',
    'SimMetricsConfig',
]


@lru_cache(maxsize=1)
def _get_package_obj():
    import mlgixd
    return mlgixd


def _default_dict(**kwargs) -> Field:
    return field(default_factory=lambda: deepcopy(kwargs))


@dataclass
class _BaseConfig:
    def __post_init__(self):
        for conf_name, conf_type in self.__annotations__.items():
            value = getattr(self, conf_name)
            if conf_type == type and isinstance(value, str):
                new_value = getattr(_get_package_obj(), value, value)
                if isinstance(new_value, type):
                    setattr(self, conf_name, new_value)

    def asdict(self):
        data_dict = asdict(self)
        for conf_name, conf_type in self.__annotations__.items():
            if isinstance(data_dict[conf_name], type):
                data_dict[conf_name] = data_dict[conf_name].__name__

        return data_dict


@dataclass
class GeneralConfig(_BaseConfig):
    name: str
    dest_path: str
    train: bool = True
    test: bool = True
    save: bool = True
    evaluate_fps: int = 1000
    save_model_size: bool = True
    disable_tqdm: bool = True

    __version__: str = __version__

    def asdict(self):
        d = super().asdict()
        d['__version__'] = __version__
        return d


@dataclass
class ModelConfig(_BaseConfig):
    model: type = 'ModifiedFasterRCNN'
    init_from_model: str = ''
    use_default: bool = True
    params: dict = _default_dict(pretrained=True)


@dataclass
class TrainerConfig(_BaseConfig):
    init_lr: float = 2e-3
    batch_size: int = 16
    num_batches: int = 3000
    lr_scheduler: type = 'ExpDecayLRCallback'
    lr_scheduler_kwargs: dict = _default_dict(gamma=0.5, freq=500)


@dataclass
class SimMetricsConfig(_BaseConfig):
    num: Union[int, List[int]] = 10000
    nms_level: Union[float, List[float]] = 0.1
    score_level: Union[float, List[float]] = 0.05
    match_threshold: Union[float, List[float]] = 1
    match_criterion: Union[str, List[str]] = 'q'
    min_iou: Union[float, List[float]] = 0.3


@dataclass
class Config(_BaseConfig):
    general: GeneralConfig
    model: ModelConfig
    training: TrainerConfig
    sim_metrics: SimMetricsConfig

    def asdict(self):
        return {name: getattr(self, name).asdict() for name in self.__annotations__.keys()}

    @classmethod
    def from_dict(cls, conf_dict: dict):
        return cls(*[
            conf_type(**conf_dict.get(conf_name, {}))
            for conf_name, conf_type in cls.__annotations__.items()
        ])

    @classmethod
    def load(cls, path):
        with open(str(path), 'r') as f:
            state_dict = yaml.load(f, Loader=yaml.SafeLoader)
        return cls.from_dict(state_dict)

    def save(self, path):
        with open(str(path), 'w') as f:
            yaml.dump(self.asdict(), f)

    @staticmethod
    def save_default(path, name: str = 'default'):
        dest_path = '.'.join(str(path).split('.')[:-1] + ['pt'])

        config = Config(
            GeneralConfig(name, dest_path=dest_path),
            ModelConfig(),
            TrainerConfig(),
            SimMetricsConfig(),
        )
        config.save(path)
