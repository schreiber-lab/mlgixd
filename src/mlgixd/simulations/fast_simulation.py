# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from math import pi
import random

import numpy as np

import torch
import torch.nn.functional as F
from torchvision.ops import nms

from .utils import (
    normalize,
    with_probability,
    hist_equalization,
)

from .perlin import perlin_noise

from .angle_limits import AngleLimits


class FastSimulation(object):
    def __init__(self, device: torch.device = 'cuda'):
        self.device = device
        self.x = torch.arange(512, device=device)[None, :, None]
        self.y = torch.arange(512, device=device)[:, None, None]

        self.mask_coords = torch.flip((self.x * torch.cos(self.y / 512 * pi / 2)).squeeze(), (0,))
        self.angle_limits = AngleLimits()

        self.kernel1 = torch.tensor(_SMOOTH_KERNEL, device=device).view(1, 1, 3, 3)

    @torch.no_grad()
    def simulate_img(self):
        # generate material-independent peak positions & intensities
        boxes, pos, widths, a_pos, a_widths = self.simulate_labels()
        intensities = gen_intensities(boxes.shape[0], pos, widths)

        # create an image with 2D-Gaussian peaks
        img = self.img_from_labels(pos, widths, a_pos, a_widths, intensities)

        # rings modification (simulate grain orientation distribution)
        img = mul_perlin(img)

        # add background
        img = background_perlin(img)
        img = add_glass(img, self.x, self.y)
        img = add_linear_background(img)

        # add noise
        img = apply_poisson_noise(img)
        img = apply_speckle_noise(img)
        img = digitalize_img(img)
        img = apply_stretch(img)
        img = apply_stretch(img, (20, 50), (7, 10))

        # add masks
        self.add_dark_area(img)
        img = add_masks(img, self.mask_coords)

        # apply kernels & contrast correction
        img = apply_kernel(img, self.kernel1)
        img = apply_he(img)
        img = apply_clip_img(img)

        # rescale intensities to [0, 1]
        img = normalize(img)

        return img, boxes

    @torch.no_grad()
    def simulate_boxes(self):
        boxes, *_ = self.simulate_labels()
        return boxes

    @torch.no_grad()
    def simulate_labels(self):
        self.angle_limits.update_params()

        pos, widths, a_pos, a_widths = simulate_labels(self.device)
        pos, widths, a_pos, a_widths = filter_nms(pos, widths, a_pos, a_widths)

        boxes = torch.stack([pos - widths,
                             a_pos - a_widths * 2,
                             pos + widths,
                             a_pos + a_widths * 2], 1)  # x0, y0, x1, y1

        boxes, indices = self.filter_dark_area(pos, boxes)

        if not boxes.shape[0]:
            return self.simulate_labels()

        clamp_boxes(boxes)

        pos, widths, a_pos, a_widths = pos[indices], widths[indices], a_pos[indices], a_widths[indices]

        return boxes, pos, widths, a_pos, a_widths

    def add_dark_area(self, img):
        dark_area_idx = (self.y <= self.angle_limits.min(self.x)) | (self.y >= self.angle_limits.max(self.x))

        img[dark_area_idx.squeeze()] = 0

    def filter_dark_area(self, pos, boxes, min_angle: float = 40):
        angles = (boxes[:, 3] + boxes[:, 1]) / 2
        boxes[:, 3] = torch.minimum(boxes[:, 3], self.angle_limits.max(pos))
        boxes[:, 1] = torch.maximum(boxes[:, 1], self.angle_limits.min(pos))

        widths = boxes[:, 3] - boxes[:, 1]

        indices = (widths >= min_angle) & (angles - boxes[:, 1] > - widths / 2) & (angles < boxes[:, 3])

        return boxes[indices], indices

    def img_from_labels(self, pos, widths, a_pos, a_widths, intensities):
        power = 2 if random.random() > 0.2 else 4
        return (intensities[None, None] * (
            torch.exp(
                - torch.abs(self.x - pos[None, None]) ** power / widths[None, None] ** power / 2
                - (self.y - a_pos[None, None]) ** 2 / a_widths[None, None] ** 2 / 2
            )
        )).sum(-1)


def get_power():
    r = random.randint(2, 4)
    if r > 0.3:
        return 2
    elif r > 0.1:
        return 3
    return 4


@with_probability(0.7)
def apply_kernel(img, kernel):
    return F.conv2d(img[None, None], kernel, padding=1).squeeze()


@with_probability(0.7)
def apply_he(img):
    return hist_equalization(img)


@with_probability(0.3)
def apply_clip_img(img):
    m, s = img.mean().item(), img.std().item() * random.uniform(2, 4)
    return torch.clamp_(img, m - s, m + s)


@with_probability(0.9)
def mul_perlin(img):
    rates = (3, 4) if random.random() > 0.3 else (3, 4, 5)
    weights = tuple(np.random.uniform(1, 3, len(rates)))
    return img * perlin_noise(rates, weights, amp=1, device=img.device)


def _bernoulli(p, shape, device):
    return torch.bernoulli(torch.empty(*shape, device=device).uniform_(0, 2 * p)) == 1


@with_probability(0.2)
def background_perlin(img):
    return img + perlin_noise((1,), device=img.device) * img.max() / 2


@with_probability(0.3)
def apply_speckle_noise(img):
    var = random.uniform(0.1, 0.25)
    noise = torch.normal(0, var, img.shape, device=img.device)
    return img + img * noise


@with_probability(1)
def apply_poisson_noise(img):
    coef = random.random() * 50 + 50
    return torch.poisson(coef * normalize(img))


@with_probability(0.9)
def digitalize_img(img):
    channels = random.randint(10, 32)
    return (normalize(img) * channels).round()


@with_probability(1)
def add_glass(img, x, y, pos_range: tuple = (40, 300)):
    power = 2 if random.random() > 0.5 else 1

    r = random.uniform(*pos_range)
    w = random.uniform(50, 140)
    a = random.uniform(50, 450)
    aw = random.uniform(250, 1050)
    weight = random.uniform(0.5, 1.2)

    if power == 1:
        weight *= 2
        w *= 2

    gauss = torch.exp(- torch.abs(x - r) ** power / 2 / w ** power - (y - a) ** 2 / 2 / aw ** 2).squeeze()
    return normalize(img) + gauss * weight


@with_probability(0.5)
def add_masks(img, coords):
    n = random.randint(1, 2)
    rs = np.random.uniform(130, 280, n)
    ws = np.random.uniform(3, 6, n)

    if n == 2 and abs(rs[1] - rs[0]) < 100:
        rs, ws = rs[:1], ws[:1]

    for r, w in zip(rs, ws):
        img[(coords <= (r + w)) & (coords >= (r - w))] = 0
    return img


@with_probability(0.9)
def add_linear_background(img):
    start, end = np.random.uniform(0, 0.1, 2)
    return normalize(img) + torch.linspace(start, end, 512, device=img.device)[None].repeat(512, 1)


@with_probability(0.8)
def apply_stretch(img, x_range: tuple = (50, 150), step_range: tuple = (3, 6)):
    x_max = random.randint(*x_range)
    step = random.randint(*step_range)
    stretched = torch.nn.functional.interpolate(
        img[::step, :x_max][None, None],
        (512, x_max)
    ).squeeze()
    img[:, :x_max] = stretched
    return img


def gen_intensities(num: int, pos, ws):
    intensities = torch.rand(num, device=pos.device) * 15 + 5
    intensities[(pos < 160) | (ws < 2)] *= 2.5
    return intensities


def filter_nms(pos, widths, a_pos, a_widths, min_nms: float = 0.001):
    idx_boxes = torch.stack([pos - widths * 2.5, a_pos - a_widths * 2, pos + widths * 2.5, a_pos + a_widths * 2], 1)
    indices = nms(idx_boxes, torch.ones(idx_boxes.shape[0], device=idx_boxes.device, dtype=torch.float), min_nms)
    return pos[indices], widths[indices], a_pos[indices], a_widths[indices]


def simulate_labels(device: torch.device = 'cuda'):
    n = random.randint(2, 200)

    width_central = random.uniform(0.8, 4.5)
    pos = torch_uniform(70, 500, n, device=device)
    widths = torch.poisson(torch.tensor([width_central] * n, device=device) * 50) / 50
    a_pos = torch_uniform(-20, 532, n, device=device)
    a_widths = torch_uniform(40, 500, n, device=device)

    return pos, widths, a_pos, a_widths


def torch_uniform(low=0, high=1, *sizes, device: torch.device = 'cuda'):
    return torch.rand(*sizes, device=device) * (high - low) + low


_SMOOTH_KERNEL = [[1., 1., 1.],
                  [1., 0.3, 1.],
                  [1., 1., 1.]]


def clamp_boxes(boxes, size: int = 512):
    torch.clamp_(boxes[:, 0], min=0)
    torch.clamp_(boxes[:, 1], min=0)
    torch.clamp_(boxes[:, 2], max=size)
    torch.clamp_(boxes[:, 3], max=size)
