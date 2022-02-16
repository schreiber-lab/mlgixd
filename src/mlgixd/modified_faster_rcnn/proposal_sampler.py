# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import Tensor

from numpy import arange
from numpy.random import choice

from .utils import valid_boxes


class ProposalSampler(object):
    def __init__(self,
                 random_anchors_per_image: int = 10,
                 num_samples_from_targets: int = 2,
                 sample_x_std: float = 1,
                 sample_y_std: float = 10,
                 ):
        self.random_anchors_per_image = random_anchors_per_image
        self.num_samples_from_targets = num_samples_from_targets
        self.std = torch.tensor([sample_x_std, sample_y_std, sample_x_std, sample_y_std])[None]

    def __call__(self, proposals: List[Tensor], targets: List[Tensor], anchors: Tensor) -> List[Tensor]:
        std = self.std.to(anchors.device)
        num_anchors = anchors.shape[0]
        num_rand = self.random_anchors_per_image
        anchor_indices = arange(num_anchors)

        training_proposals: List[Tensor] = []

        for i, (proposals_per_image, targets_per_image) in enumerate(zip(proposals, targets)):
            training_proposals_per_image = [proposals_per_image, targets_per_image]

            if num_rand:
                indices = choice(anchor_indices, num_rand, replace=False)
                training_proposals_per_image.append(anchors[indices])

            for _ in range(self.num_samples_from_targets):
                sampled = torch.normal(targets_per_image, std)
                keep = valid_boxes(sampled)

                training_proposals_per_image.append(sampled[keep])

            training_proposals.append(torch.cat(training_proposals_per_image))

        return training_proposals
