# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from .unet_backbone import UNetBackbone
from .segmentation_adapter import SegmentationAdapter
from ..ml import ModelMixin


class UNet(SegmentationAdapter, ModelMixin):
    def __init__(self, thresh: float = 0.5, **kwargs):
        super().__init__(UNetBackbone(), thresh, **kwargs)

    @classmethod
    def default_model(cls, *args, **kwargs):
        return cls().cuda()
