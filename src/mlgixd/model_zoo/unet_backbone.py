# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

# Mostly copied from https://github.com/zhoudaxia233/PyTorch-Unet


import torch
import torch.nn as nn

from ..ml import init_kaiming


class UNetBackbone(nn.Module):
    def __init__(self, out_channels: int = 1):
        super().__init__()

        self.conv1 = _double_conv(1, 64)
        self.conv2 = _double_conv(64, 128)
        self.conv3 = _double_conv(128, 256)
        self.conv4 = _double_conv(256, 512)
        self.bottleneck = _double_conv(512, 1024)
        self.up_conv5 = _up_conv(1024, 512)
        self.conv5 = _double_conv(1024, 512)
        self.up_conv6 = _up_conv(512, 256)
        self.conv6 = _double_conv(512, 256)
        self.up_conv7 = _up_conv(256, 128)
        self.conv7 = _double_conv(256, 128)
        self.up_conv8 = _up_conv(128, 64)
        self.conv8 = _double_conv(128, 64)
        self.conv9 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.sigmoid = nn.Sigmoid()

        init_kaiming(self, nonlinearity='relu')

    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.maxpool(conv1)

        conv2 = self.conv2(x)
        x = self.maxpool(conv2)

        conv3 = self.conv3(x)
        x = self.maxpool(conv3)

        conv4 = self.conv4(x)
        x = self.maxpool(conv4)

        bottleneck = self.bottleneck(x)

        x = self.up_conv5(bottleneck)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv5(x)

        x = self.up_conv6(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv8(x)

        x = self.conv9(x)
        x = self.sigmoid(x)

        return x


def _double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),  # added BN layers compared with the vanilla Unet
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def _up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )
