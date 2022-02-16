# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import remove_small_boxes

from skimage.measure import label as separate_peaks
import numpy as np


class SegmentationAdapter(nn.Module):
    def __init__(self, model: nn.Module, thresh: float = 0.5, train_shape: tuple = (512, 512)):
        super().__init__()
        self.thresh = thresh
        self.model = model
        self.shape = train_shape
        self.x = torch.arange(train_shape[1])[None, :, None]
        self.y = torch.arange(train_shape[0])[:, None, None]
        self.criterion = nn.BCELoss()

    def forward(self, imgs, targets=None):
        if self.training:
            return self.loss(imgs, targets)
        else:
            return self.get_prediction_boxes(imgs)

    @torch.no_grad()
    def get_prediction_boxes(self, imgs):
        pred_masks = self.model(imgs)

        boxes, scores = zip(*[self.pred_mask_to_boxes(pred_mask) for pred_mask in pred_masks])

        return boxes, scores

    def pred_mask_to_boxes(self, pred_mask: Tensor):
        peak_mask = (pred_mask.detach()[0] > self.thresh).to(torch.float)

        return self._peak_mask_to_boxes(peak_mask)

    def _peak_mask_to_boxes(self, peak_mask: Tensor):
        peak_indices = separate_peaks(peak_mask.cpu().numpy())

        boxes_list = []
        scores = []

        for peak_idx in range(1, peak_indices.max()):
            row_idx, col_idx = np.where(peak_indices == peak_idx)
            boxes_list.append([col_idx.min(), row_idx.min(), col_idx.max(), row_idx.max()])  # x0, y0, x1, y1
            scores.append((peak_mask[row_idx, col_idx].mean() - self.thresh) / (1 - self.thresh))  # rescale to [0, 1]

        boxes = torch.tensor(boxes_list, dtype=torch.float, device=peak_mask.device).view(-1, 4)
        scores = torch.atleast_1d(torch.tensor(scores, dtype=torch.float, device=peak_mask.device))

        keep = remove_small_boxes(boxes, 1e-2)
        boxes, scores = boxes[keep], scores[keep]

        return boxes, scores

    def loss(self, imgs, targets) -> dict:
        masks_pred = self.model(imgs)
        true_masks = self.targets_to_masks(targets)
        loss = self.criterion(masks_pred, true_masks)
        return {
            'cross_entropy_loss': loss
        }

    @torch.no_grad()
    def targets_to_masks(self, targets):
        return torch.stack([self.boxes_to_masks(boxes) for boxes in targets], 0)

    def boxes_to_masks(self, boxes: Tensor):
        x0, y0, x1, y1 = boxes.T
        x, y = self.x.to(boxes), self.y.to(boxes)

        true_mask = (
                (x0[None, None] < x) &
                (x1[None, None] > x) &
                (y0[None, None] < y) &
                (y1[None, None] > y)
        ).sum(-1).to(torch.float)[None]

        return true_mask
