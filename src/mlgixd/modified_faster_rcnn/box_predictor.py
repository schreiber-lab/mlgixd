# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn

__all__ = [
    'TwoMLPHead',
    'FastRCNNPredictor'
]


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size: int = 1024, slope: float = 0.01):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.leaky_relu = nn.LeakyReLU(slope, inplace=True)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = self.leaky_relu(self.fc6(x))
        x = self.leaky_relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, 1)
        self.bbox_pred = nn.Linear(in_channels, 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]

        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
