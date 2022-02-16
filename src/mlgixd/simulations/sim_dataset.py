# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .fast_simulation import FastSimulation


class SimDataset(object):
    def __init__(self, sim: FastSimulation = None, *, in_channels: int = 1):
        self.sim = sim or FastSimulation()
        self.in_channels = in_channels

    def get_batch(self, size: int):
        images, boxes = [], []

        for _ in range(size):
            img, bx = self.sim.simulate_img()
            images.append(img)
            boxes.append(bx)

        images = torch.stack(images)[:, None]

        if self.in_channels > 1:
            images = images.repeat(1, self.in_channels, 1, 1)

        return images, boxes
