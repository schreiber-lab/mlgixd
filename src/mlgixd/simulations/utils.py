# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch
from torch import Tensor


@torch.no_grad()
def hist_equalization(img: Tensor, bins: int = 1000):
    bin_edges = torch.linspace(img.min(), img.max(), bins + 1, device=img.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    img_flat = img.view(-1)
    hist = torch.histc(img_flat, bins=bins)
    cdf = torch.cumsum(hist, 0)
    cdf = cdf / cdf[-1]
    res = interp1d(bin_centers, cdf, img_flat)
    return res.view(img.shape)


def interp1d(x: Tensor, y: Tensor, x_new: Tensor) -> Tensor:
    eps = torch.finfo(y.dtype).eps
    ind = torch.searchsorted(x.contiguous(), x_new.contiguous())
    ind = torch.clamp(ind - 1, 0, x.shape[0] - 2)
    slopes = (y[1:] - y[:-1]) / (eps + (x[1:] - x[:-1]))
    return y[ind] + slopes[ind] * (x_new - x[ind])


def with_probability(probability: float):
    def wrapper(func):
        def new_func(img, *args, **kwargs):
            return func(img, *args, **kwargs) if random.random() < probability else img

        return new_func

    return wrapper


def normalize(img: Tensor) -> Tensor:
    return (img - img.min()) / (img.max() - img.min())
