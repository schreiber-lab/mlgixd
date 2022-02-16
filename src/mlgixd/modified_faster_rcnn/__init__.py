# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.
# ------------------------------------------------------------------------
# Modified from torchvision (https://github.com/pytorch/vision/blob/main/torchvision/models/detection)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .utils import (
    BalancedPositiveNegativeSampler,
    Matcher,
)
from .backbone import VCompressResNet
from .filter_proposals import FilterProposals
from .fixed_anchors import FixedAnchorsGenerator
from .rpn_head import RPNHead
from .encode_boxes import (
    encode_boxes,
    decode_boxes,
)
from .modifiedfasterrcnn import (
    ModifiedFasterRCNN,
    TwoMLPHead,
    FastRCNNPredictor,
    RoiAlign
)

from .one_stage_detector import RPN

from .utils import (
    assign_targets_to_anchors
)

from .encode_boxes import (
    encode_boxes,
    decode_boxes
)

__all__ = [
    'VCompressResNet',
    'encode_boxes',
    'decode_boxes',
    'RPNHead',
    'FixedAnchorsGenerator',
    'BalancedPositiveNegativeSampler',
    'Matcher',
    'assign_targets_to_anchors',
    'FilterProposals',
    'ModifiedFasterRCNN',
    'RPN',
    'TwoMLPHead',
    'FastRCNNPredictor',
    'RoiAlign',
]
