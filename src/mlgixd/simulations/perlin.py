# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import torch

__all__ = ['perlin_noise', 'perlin_octave']


def perlin_noise(octave_rates: tuple = (1, 2, 3, 4),
                 weights: tuple = None,
                 amp: float = 1., size: int = 512, device='cuda'):
    weights = weights or [1] * len(octave_rates)

    p = 0

    for rate, weight in zip(octave_rates, weights):
        octave = 2 ** rate
        p += perlin_octave(octave, octave, size // octave, device=device) * weight
    return ((p - p.min()) / (p.max() - p.min()) - 0.5) * amp + 1


def perlin_octave(width, height, scale, device='cuda'):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)

    wx = 1 - interp(xs)
    wy = 1 - interp(ys)

    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))

    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def interp(t):
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
