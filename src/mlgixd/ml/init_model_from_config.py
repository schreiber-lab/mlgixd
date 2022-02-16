# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from ..config import Config
from .model_mixin import ModelMixin, ModelType


def init_model_from_config(config: Config) -> ModelType:
    model_cls = config.model.model

    if not isinstance(model_cls, type(ModelMixin)):
        raise ValueError(f"Unexpected model class {model_cls}")

    if config.model.use_default:
        model_cls = model_cls.default_model
    model = model_cls(**config.model.params)
    if config.model.init_from_model:
        model.load_model(config.model.init_from_model)
    return model
