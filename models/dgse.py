"""Depth-Guided Geometric-Spatial Prior Extractor (DGSE).

Operates on the Macro-Pixel Image (MacPI) representation so that spatial-angular
correlations are explicitly organized. The derained depth D_Mac is concatenated
with I_Mac and M_Mac, and used as a slope prior to gate the extracted
geometry-spatial features F_Mac.
"""

import torch
import torch.nn as nn

from utils.misc import _ConvBnReLU, ResBlock


class DGSE(nn.Module):
    def __init__(self, in_ch=5, base_ch=64):
        super().__init__()
        self.stem = _ConvBnReLU(in_ch, base_ch, 3, 1)
        self.blocks = nn.Sequential(*[ResBlock(base_ch) for _ in range(4)])
        self.depth_proj = nn.Sequential(
            nn.Conv2d(1, base_ch, 1), nn.Sigmoid())

    def forward(self, d_mac, i_mac, m_mac):
        """
        d_mac: (B, 1, U*H, V*W) derained depth in MacPI
        i_mac: (B, 3, U*H, V*W) rainy image in MacPI
        m_mac: (B, 1, U*H, V*W) rain mask in MacPI
        returns: F_Mac (B, base_ch, U*H, V*W)
        """
        x = torch.cat([d_mac, i_mac, m_mac], dim=1)
        f = self.stem(x)
        f = self.blocks(f)
        # depth acts as a slope prior aligning epipolar trajectories
        gate = self.depth_proj(d_mac)
        f = f * gate + f
        return f
