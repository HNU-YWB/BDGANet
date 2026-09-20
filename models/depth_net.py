"""Geometry reconstructor: rain-free depth restoration network.

Backbone follows the resolution-adaptive encoder-decoder design of RADepth [8]:
an encoder extracts multi-scale features from the concatenation of the rainy
depth prior D_occ, rain mask M_occ and rainy image I_occ, and a decoder
progressively upsamples them into a rain-free depth map D_de.
"""

import torch
import torch.nn as nn

from utils.misc import _ConvBnReLU, ResBlock, DecBlock


class GeometryReconstructor(nn.Module):
    def __init__(self, in_ch=5, base_ch=32):
        super().__init__()
        self.stem = _ConvBnReLU(in_ch, base_ch, 3, 1)
        self.enc1 = ResBlock(base_ch, stride=2)          # H/2
        self.enc2 = nn.Sequential(
            _ConvBnReLU(base_ch, base_ch * 2, 3, 2), ResBlock(base_ch * 2))  # H/4
        self.enc3 = nn.Sequential(
            _ConvBnReLU(base_ch * 2, base_ch * 4, 3, 2), ResBlock(base_ch * 4))  # H/8
        self.enc4 = nn.Sequential(
            _ConvBnReLU(base_ch * 4, base_ch * 8, 3, 2), ResBlock(base_ch * 8))  # H/16

        self.dec4 = DecBlock(base_ch * 8, base_ch * 8, scale=2)
        self.dec3 = DecBlock(base_ch * 8 + base_ch * 4, base_ch * 4, scale=2)
        self.dec2 = DecBlock(base_ch * 4 + base_ch * 2, base_ch * 2, scale=2)
        self.dec1 = DecBlock(base_ch * 2 + base_ch, base_ch, scale=2)
        self.head = nn.Sequential(
            _ConvBnReLU(base_ch * 2, base_ch // 2, 3, 1),
            nn.Conv2d(base_ch // 2, 1, 3, 1, 1))

    def forward(self, x):
        """x: (B, 5, H, W) -> (B, 1, H, W)"""
        e0 = self.stem(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))
        out = self.head(torch.cat([d1, e0], 1))
        return out
