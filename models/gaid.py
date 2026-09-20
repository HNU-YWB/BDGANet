"""Geometry-Appearance Interactive Derainer (GAID).

Contains the geometry reconstructor, the depth-guided appearance reconstructor
(DGSE + AVAA + encoder-decoder), and the Bidirectional Difference-Perception
mechanism that couples the two branches during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import _ConvBnReLU, ResBlock, DecBlock, sai_to_macpi, macpi_to_sai
from .depth_net import GeometryReconstructor
from .dgse import DGSE
from .avaa import AVAA


class GAID(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, angular_u=5, angular_v=5,
                 center_view=12, depth_ch=32, bottleneck=16):
        super().__init__()
        self.uv = angular_u * angular_v
        self.center = center_view
        self.angular_u, self.angular_v = angular_u, angular_v

        self.geo = GeometryReconstructor(in_ch=in_ch + 2, base_ch=depth_ch)  # [D, M, I]
        self.dgse = DGSE(in_ch=in_ch + 2, base_ch=base_ch)
        self.avaa = AVAA(in_ch=base_ch, bottleneck=bottleneck)

        # appearance encoder-decoder (center view branch)
        self.stem = _ConvBnReLU(in_ch, base_ch, 3, 1)
        self.enc1 = nn.Sequential(
            _ConvBnReLU(base_ch, base_ch * 2, 3, 2), ResBlock(base_ch * 2))   # H/2
        self.enc2 = nn.Sequential(
            _ConvBnReLU(base_ch * 2, base_ch * 4, 3, 2), ResBlock(base_ch * 4))  # H/4
        self.enc3 = nn.Sequential(
            _ConvBnReLU(base_ch * 4, base_ch * 8, 3, 2), ResBlock(base_ch * 8))  # H/8
        self.dec3 = DecBlock(base_ch * 8 + base_ch * 4, base_ch * 4, scale=2)
        self.dec2 = DecBlock(base_ch * 4 + base_ch * 2, base_ch * 2, scale=2)
        self.dec1 = DecBlock(base_ch * 2 + base_ch, base_ch, scale=2)
        self.head = nn.Conv2d(base_ch, in_ch, 3, 1, 1)

        # appearance-difference perception map W_occ
        self.dp_proj = nn.Sequential(
            _ConvBnReLU(in_ch, 32, 3, 1), nn.Conv2d(32, 1, 1))

    def forward(self, i_occ, d_occ, m_occ, i_gt_c=None, d_gt_c=None, mode='infer'):
        """
        i_occ: (B, UV, 3, H, W)
        d_occ: (B, UV, 1, H, W) rainy-scene depth prior
        m_occ: (B, UV, 1, H, W) rain mask
        i_gt_c / d_gt_c: center-view ground truth (required in 'train' mode)

        returns dict: i_out, d_de (UV), d_de_c, m_occ, and in train mode d_de_r, w_occ
        """
        b, uv, c, h, w = i_occ.shape

        # ---- geometry reconstructor: rain-free depth (per view) ----
        geo_in = torch.cat([d_occ, m_occ, i_occ], dim=2).view(b * uv, c + 2, h, w)
        d_de = self.geo(geo_in).view(b, uv, 1, h, w)
        d_de_c = d_de[:, self.center]

        # ---- DGSE on MacPI ----
        d_mac = sai_to_macpi(d_de, self.angular_u, self.angular_v)
        i_mac = sai_to_macpi(i_occ, self.angular_u, self.angular_v)
        m_mac = sai_to_macpi(m_occ, self.angular_u, self.angular_v)
        f_mac = self.dgse(d_mac, i_mac, m_mac)                       # (B, C, U*H, V*W)

        # ---- AVAA: adaptive multi-view aggregation ----
        f_sai = macpi_to_sai(f_mac, self.angular_u, self.angular_v)  # (B, UV, C, H, W)
        f_ref = self.avaa(f_sai, self.center)                        # (B, C, H, W)

        # ---- appearance encoder-decoder (center view, residual learning) ----
        e0 = self.stem(i_occ[:, self.center]) + f_ref
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(torch.cat([e3, F.interpolate(e2, size=e3.shape[-2:], mode='bilinear', align_corners=False)], 1))
        d2 = self.dec2(torch.cat([d3, F.interpolate(e1, size=d3.shape[-2:], mode='bilinear', align_corners=False)], 1))
        d1 = self.dec1(torch.cat([d2, F.interpolate(e0, size=d2.shape[-2:], mode='bilinear', align_corners=False)], 1))
        i_out = i_occ[:, self.center] + self.head(d1)

        outs = {'i_out': i_out, 'd_de': d_de, 'd_de_c': d_de_c, 'm_occ': m_occ}

        if mode == 'train':
            assert i_gt_c is not None and d_gt_c is not None
            # geometry-difference perception: refine depth from appearance output
            d_de_r = self.geo(torch.cat([i_out, d_de_c, d_gt_c], dim=1))
            outs['d_de_r'] = d_de_r
            # appearance-difference perception map
            a_d = i_out - i_gt_c
            w_occ = torch.softmax(self.dp_proj(a_d).flatten(1), dim=1).view(b, 1, h, w)
            outs['w_occ'] = w_occ
        return outs
