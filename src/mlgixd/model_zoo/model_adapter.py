# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import torch


class ModelAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.cuda()

    def forward(self, imgs, targets=None):
        imgs = imgs.repeat(1, 3, 1, 1)
        if targets is not None:
            targets = _convert_targets_to_dicts(targets)
        res = self.model(imgs, targets)

        if self.training:
            return res

        boxes = [r['boxes'] for r in res]
        scores = [r['scores'] for r in res]

        return boxes, scores


def _convert_targets_to_dicts(targets):
    return [
        {
            'boxes': target, 'labels': torch.ones(target.shape[0], device=target.device, dtype=torch.long)
        }
        for target in targets
    ]
