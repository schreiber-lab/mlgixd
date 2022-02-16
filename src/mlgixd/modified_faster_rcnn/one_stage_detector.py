# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List

from torchvision.ops import box_iou
from torch import Tensor

from .modifiedfasterrcnn import (
    ModifiedFasterRCNN,
    BackboneType,
    TransformImg,
    Matcher,
    BalancedPositiveNegativeSampler,
    RPNHead,
    FilterProposals,
    FixedAnchorsGenerator,
    BackboneWithFPN,
    VCompressResNet,
)


class RPN(ModifiedFasterRCNN):
    def __init__(self, backbone: 'BackboneType',
                 height_weight_per_feature: Tuple[Tuple[Tuple[float, float], ...], ...] = None,
                 img_shape: Tuple[int, int] = (512, 512),
                 transform_img: 'TransformImg' = None,
                 rpn_matcher: 'Matcher' = None,
                 rpn_sampler: 'BalancedPositiveNegativeSampler' = None,
                 rpn_filter: 'FilterProposals' = None,
                 rpn_head: 'RPNHead' = None,
                 anchor_generator: 'FixedAnchorsGenerator' = None,
                 rpn_reg_weight: float = 10.,
                 rpn_objectness_weight: float = 1.,
                 loss_func=None,
                 use_box_padding: bool = False,
                 rpn_box_similarity=box_iou,
                 ):

        super().__init__(
            backbone=backbone,
            height_weight_per_feature=height_weight_per_feature,
            img_shape=img_shape,
            transform_img=transform_img,
            rpn_matcher=rpn_matcher,
            rpn_sampler=rpn_sampler,
            rpn_filter=rpn_filter,
            use_box_padding=use_box_padding,
            proposal_sampler=1,  # bool(1) == True
            roi_matcher=1,
            roi_filter=1,
            rpn_head=rpn_head,
            anchor_generator=anchor_generator,
            roi_align=1,
            box_head=1,
            box_predictor=1,
            rpn_reg_weight=rpn_reg_weight,
            rpn_objectness_weight=rpn_objectness_weight,
            loss_func=loss_func,
            train_rpn=True,
            train_roi=False,
            rpn_box_similarity=rpn_box_similarity,
        )

    @classmethod
    def default_model(
            cls,
            name: str = None,
            pretrained: bool = True,
    ):

        rpn_filter = FilterProposals(
            score_thresh=0.1,
            post_nms_top_n_test=300,
            nms_thresh=0.5,
        )

        backbone = BackboneWithFPN(
            VCompressResNet(
                channels=(64, 128, 256, 256),
                include_features_list=(2, 3, 4),
                init_from_resnet=pretrained,
            ),
            backbone_channels=(128, 256, 256),
            out_channels=64,
        )

        model = cls(
            backbone,
            height_weight_per_feature=(
                ((50, 10), (100, 10)),
                ((200, 10), (300, 10)),
                ((400, 10), (500, 10)),
            ),
            rpn_reg_weight=10,
            rpn_box_similarity=box_iou,
            rpn_filter=rpn_filter,
        ).cuda()

        if name:
            model.load_model(name)

        return model

    def forward(self, imgs: Tensor, targets: List[Tensor] = None):
        if self.training:
            return super().forward(imgs, targets)
        return self.get_rpn_proposals(imgs)
