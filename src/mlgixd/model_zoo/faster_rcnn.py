# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import torchvision

from .model_adapter import ModelAdapter
from ..ml import ModelMixin


class FasterRCNN(ModelAdapter, ModelMixin):
    @classmethod
    def default_model(cls, pretrained: bool = True):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=pretrained, num_classes=2)
        return cls(model)
