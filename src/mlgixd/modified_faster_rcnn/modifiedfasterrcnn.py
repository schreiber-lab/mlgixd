# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List, Dict, Callable

import torch
from torch import nn
from torch import Tensor

from torchvision.ops import box_iou

from ..ml import ModelMixin

from .rpn_head import RPNHead
from .fixed_anchors import FixedAnchorsGenerator
from .encode_boxes import decode_boxes
from .utils import (
    BalancedPositiveNegativeSampler,
    Matcher
)
from .backbone import (
    BackboneType,
    VCompressResNet,
)
from .filter_proposals import FilterProposals
from .filter_rois import FilterRois
from .transform_img import TransformImg
from .losses import (
    calc_losses
)

from .roi_align_layer import RoiAlign, ChooseOneMap
from .box_predictor import (
    FastRCNNPredictor,
    TwoMLPHead
)
from .transform_boxes import BoxWidthPadding
from .proposal_sampler import ProposalSampler
from .transform_list import TransformList

from .fpn import BackboneWithFPN


class ModifiedFasterRCNN(nn.Module, ModelMixin):
    def __init__(self,
                 backbone: BackboneType,
                 height_weight_per_feature: Tuple[Tuple[Tuple[float, float], ...], ...] = None,
                 img_shape: Tuple[int, int] = (512, 512),
                 transform_img: TransformImg = None,
                 rpn_matcher: Matcher = None,
                 roi_matcher: Matcher = None,
                 rpn_sampler: BalancedPositiveNegativeSampler = None,
                 roi_sampler: BalancedPositiveNegativeSampler = None,
                 rpn_filter: FilterProposals = None,
                 roi_filter: FilterRois = None,
                 rpn_head: RPNHead = None,
                 anchor_generator: FixedAnchorsGenerator = None,
                 roi_align: RoiAlign = None,
                 box_head: TwoMLPHead = None,
                 box_predictor: FastRCNNPredictor = None,
                 max_num_per_image: int = 250,
                 representation_size: int = 1024,
                 rpn_reg_weight: float = 10.,
                 rpn_objectness_weight: float = 1.,
                 box_weight: float = 10.,
                 box_score_weight: float = 1.,
                 loss_func=None,
                 train_rpn: bool = True,
                 train_roi: bool = True,
                 rpn_target_transforms: List[Callable[[List[Tensor]], List[Tensor]]] = (),
                 use_box_padding: bool = True,
                 proposal_sampler: ProposalSampler = None,
                 rpn_box_similarity=box_iou,
                 roi_box_similarity=box_iou,
                 ):
        super().__init__()

        self.loss_weights = {
            'rpn_objectness': rpn_objectness_weight,
            'rpn_box_reg': rpn_reg_weight,
            'score': box_score_weight,
            'box_reg': box_weight
        }

        self.rpn_box_similarity = rpn_box_similarity
        self.roi_box_similarity = roi_box_similarity

        self.train_rpn = train_rpn
        self.train_roi = train_roi

        self.rpn_target_transforms: TransformList = _init_rpn_transformations(rpn_target_transforms, use_box_padding)

        self.loss_func = loss_func or calc_losses

        self.transform_img = transform_img or TransformImg()

        self.backbone = backbone

        self.rpn_sampler = rpn_sampler or BalancedPositiveNegativeSampler(
            batch_size_per_image=256, positive_fraction=0.5)
        self.roi_sampler = roi_sampler or BalancedPositiveNegativeSampler(
            batch_size_per_image=128, positive_fraction=0.5)

        self.proposal_sampler = proposal_sampler or ProposalSampler()

        self.rpn_matcher = rpn_matcher or Matcher(
            high_threshold=0.34, low_threshold=0.15, allow_low_quality_matches=True
        )
        self.roi_matcher = roi_matcher or Matcher(
            high_threshold=0.3, low_threshold=0.01, allow_low_quality_matches=False
        )

        self._img_shape = img_shape

        self.rpn_filter = rpn_filter or FilterProposals(
            pre_nms_top_n_train=2000,
            pre_nms_top_n_test=1000,
            post_nms_top_n_train=2000,
            post_nms_top_n_test=1000,
            nms_thresh=0.9,
            score_thresh=0.,
            min_size=1e-3,
            img_shape=img_shape
        )

        self.roi_filter = roi_filter or FilterRois(
            nms_thresh=0.5,
            score_thresh=0.05,
            min_size=1e-3,
            max_num_per_image=max_num_per_image,
            img_shape=img_shape
        )

        self.anchor_generator = anchor_generator or FixedAnchorsGenerator(
            height_weight_per_feature=height_weight_per_feature,
            img_shape=img_shape,
            feature_map_sizes=backbone.feature_map_sizes(img_shape),
        )

        self.rpn_head = rpn_head or RPNHead(
            backbone.out_channels,
            self.anchor_generator.num_anchors_per_location()
        )

        self.roi_align = roi_align or RoiAlign()

        self.box_head = box_head or TwoMLPHead(self.roi_align.size * backbone.out_channels, representation_size)
        self.box_predictor = box_predictor or FastRCNNPredictor(representation_size)

    @classmethod
    def default_model(
            cls,
            name: str = None,
            pretrained: bool = True,
    ):
        backbone = BackboneWithFPN(
            VCompressResNet(
                channels=(64, 128, 256, 256),
                include_features_list=(2, 3, 4),
                init_from_resnet=pretrained,
            ),
            backbone_channels=(128, 256, 256),
            out_channels=64
        )

        roi_align = RoiAlign(
            height=16, width=16,
            choose_map=ChooseOneMap(0, 3),
            feature_map_sizes=backbone.feature_map_sizes((512, 512)),
        )

        model = cls(
            backbone,
            height_weight_per_feature=(
                ((50, 10), (100, 10)),
                ((200, 10), (300, 10)),
                ((400, 10), (500, 10)),
            ),
            representation_size=32,
            roi_align=roi_align,
        ).cuda()

        if name:
            model.load_model(name)

        return model

    @property
    def img_shape(self):
        return self._img_shape

    @img_shape.setter
    def img_shape(self, shape: Tuple[int, int]):
        shape = tuple(shape)
        if shape == self._img_shape:
            return
        self._img_shape = shape
        self.anchor_generator.set_img_shape(shape, self.backbone.feature_map_sizes(shape))
        self.rpn_filter.img_shape = shape
        self.roi_filter.img_shape = shape

    def get_rpn_proposals(self, imgs: Tensor, apply_filter: bool = True):
        return self._get_proposals(imgs, apply_rpn_filter=apply_filter, get_rpn_proposals=True)

    def get_roi_proposals(self, imgs: Tensor,
                          apply_rpn_filter: bool = True,
                          apply_roi_filter: bool = True,
                          proposals: List[Tensor] or Tensor = None):
        if isinstance(proposals, Tensor):
            proposals = [proposals]
        return self._get_proposals(imgs, apply_rpn_filter, apply_roi_filter, False, proposals)

    @torch.no_grad()
    def _get_proposals(self,
                       imgs: Tensor,
                       apply_rpn_filter: bool = True,
                       apply_roi_filter: bool = True,
                       get_rpn_proposals: bool = False,
                       proposals: List[Tensor] = None
                       ):

        assert not (get_rpn_proposals and bool(proposals))

        num_images: int = imgs.shape[0]

        self.img_shape = imgs.shape[-2:]

        imgs = self.transform_img(imgs)

        features: List[Tensor] = self.backbone(imgs)

        if not proposals:

            anchors = self.anchor_generator(num_images, imgs.device)

            objectness, bbox_reg, num_anchors_per_level = self.rpn_head(features)

            rpn_proposals: Tensor = decode_boxes(bbox_reg, torch.cat(anchors)).view(num_images, -1, 4)

            if apply_rpn_filter:
                proposals, scores = self.rpn_filter(rpn_proposals, objectness, num_anchors_per_level)
            else:
                proposals, scores = rpn_proposals.split(1), torch.sigmoid(objectness).split(1)
                proposals = [p.view(-1, 4) for p in proposals]
                scores = [s.view(-1) for s in scores]

            if get_rpn_proposals:
                return proposals, scores

        num_boxes_per_image = [p.shape[0] for p in proposals]

        box_features = self.roi_align(features, proposals)
        box_features = self.box_head(box_features)
        scores, box_regression = self.box_predictor(box_features)

        predicted_boxes = decode_boxes(box_regression, torch.cat(proposals))

        if apply_roi_filter:
            boxes, scores = self.roi_filter(predicted_boxes, scores.view(-1), num_boxes_per_image)
        else:
            boxes, scores = predicted_boxes.split(num_boxes_per_image), scores.view(-1).split(num_boxes_per_image)
        return boxes, scores

    def forward(self,
                imgs: Tensor,
                targets: List[Tensor] = None,
                ):
        num_images: int = imgs.shape[0]

        self.img_shape = imgs.shape[-2:]

        losses: Dict[str, Tensor] = {}

        assert num_images > 0
        if self.training:
            assert targets is not None

        imgs = self.transform_img(imgs)

        features: List[Tensor] = self.backbone(imgs)

        anchors: List[Tensor] = self.anchor_generator(num_images, imgs.device)

        objectness, bbox_reg, num_anchors_per_level = self.rpn_head(features)

        if self.training and self.train_rpn:
            rpn_targets = self.rpn_target_transforms(targets)
            losses.update(self.calc_rpn_losses(anchors, rpn_targets, objectness, bbox_reg))
            if not self.train_roi:
                return losses

        # detach proposals!
        rpn_proposals: Tensor = decode_boxes(bbox_reg.detach(), torch.cat(anchors)).view(num_images, -1, 4)
        proposals: List[Tensor]
        roi_scores: List[Tensor]

        proposals, roi_scores = self.rpn_filter(rpn_proposals, objectness.detach(), num_anchors_per_level)

        if self.training:
            proposals = self.proposal_sampler(proposals, targets, anchors[0])

        num_boxes_per_image: List[int] = [p.shape[0] for p in proposals]

        box_features: Tensor = self.roi_align(features, proposals)
        box_features = self.box_head(box_features)
        scores, box_regression = self.box_predictor(box_features)

        if self.training:
            losses.update(self.calc_roi_losses(proposals, targets, scores, box_regression))
            return losses

        predicted_boxes = decode_boxes(box_regression, torch.cat(proposals))

        boxes, scores = self.roi_filter(predicted_boxes, scores.view(-1), num_boxes_per_image)

        return boxes, scores

    def calc_rpn_losses(self,
                        anchors: List[Tensor],
                        targets: List[Tensor],
                        objectness: Tensor,
                        bbox_reg: Tensor) -> Dict[str, Tensor]:

        rpn_objectness_loss, rpn_box_reg_loss = self.loss_func(
            self.rpn_matcher, self.rpn_sampler,
            anchors, targets, objectness, bbox_reg,
            self.rpn_box_similarity
        )
        losses: Dict[str, Tensor] = {
            'rpn_objectness': rpn_objectness_loss * self.loss_weights['rpn_objectness'],
            'rpn_box_reg': rpn_box_reg_loss * self.loss_weights['rpn_box_reg']
        }
        return losses

    def calc_roi_losses(self,
                        proposals: List[Tensor],
                        targets: List[Tensor],
                        scores: Tensor,
                        bbox_reg: Tensor
                        ):
        score_loss, box_reg_loss = self.loss_func(
            self.roi_matcher, self.roi_sampler,
            proposals, targets, scores, bbox_reg,
            self.roi_box_similarity
        )
        losses: Dict[str, Tensor] = {
            'score': score_loss * self.loss_weights['score'],
            'box_reg': box_reg_loss * self.loss_weights['box_reg']
        }
        return losses


def _init_rpn_transformations(rpn_target_transformations: list, use_box_padding: bool):
    transforms = list(rpn_target_transformations)
    if use_box_padding:
        transforms.insert(0, BoxWidthPadding())
    return TransformList(transforms)
