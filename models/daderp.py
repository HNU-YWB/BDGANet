"""Depth-Assisted Rain Edge-Region Predictor (DA-RERP).

Estimates the rain mask M_occ from rainy SAIs, Canny edges and the rainy-scene
depth prior, following the paper:
  - adaptive-threshold rain prediction -> rain probability map M_opm
  - Boundary-Aware Edge Enhancement (BAEE) via reverse attention
  - Depth-Guided Dynamic Region Enhancement (DDRE) via group-wise modulation
  - depth-guided filtering and a mask head -> M_occ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import _ConvBnReLU


class DADERP(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, use_canny=True):
        super().__init__()
        self.use_canny = use_canny
        c0 = in_ch + (1 if use_canny else 0)

        # multi-scale mask encoder (F_m1: H/2, F_m2: H/4, F_h: H/8)
        self.stem = nn.Sequential(
            _ConvBnReLU(c0, base_ch // 2, 3, 1),
            _ConvBnReLU(base_ch // 2, base_ch, 3, 1),
        )
        self.stage1 = nn.Sequential(
            _ConvBnReLU(base_ch, base_ch, 3, 2), _ConvBnReLU(base_ch, base_ch))
        self.stage2 = nn.Sequential(
            _ConvBnReLU(base_ch, base_ch, 3, 2), _ConvBnReLU(base_ch, base_ch))
        self.stage3 = nn.Sequential(
            _ConvBnReLU(base_ch, base_ch * 2, 3, 2), _ConvBnReLU(base_ch * 2, base_ch * 2))

        # adaptive-threshold rain probability prediction (M_opm)
        self.opm_conv = nn.Sequential(
            _ConvBnReLU(base_ch, base_ch // 2, 3, 1), nn.Conv2d(base_ch // 2, 1, 1))
        self.tau = nn.Parameter(torch.tensor(0.5))  # learnable threshold

        # BAEE (reverse attention on H/4 features)
        self.edge_conv = nn.Conv2d(2, 1, 1)
        self.phi = nn.Conv2d(base_ch, base_ch, 1)
        self.psi = nn.Conv2d(base_ch, base_ch, 1)
        self.fuse_conv = _ConvBnReLU(base_ch, base_ch, 3, 1)

        # DDRE (1x1 alignment -> BN -> 4-channel modulation maps)
        self.align_ee = nn.Conv2d(base_ch, base_ch, 1)
        self.align_h = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.mm_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 1),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 4, 1))
        self.hf_conv = nn.Conv2d(base_ch * 2, 64, 1)  # 4 groups x 16ch

        # depth-guided filtering gate
        self.depth_gate = nn.Sequential(
            nn.Conv2d(1, base_ch // 2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 32, 1), nn.Sigmoid())

        # mask head: H/4 -> H
        self.mask_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            _ConvBnReLU(32, base_ch, 3, 1),
            nn.Conv2d(base_ch, 1, 3, 1, 1))

    def forward(self, x, depth, edge=None):
        """
        x:     (B, UV, 3, H, W) rainy SAI volume
        depth: (B, UV, 1, H, W) rainy-scene depth prior
        edge:  (B, UV, 1, H, W) Canny edges (optional if use_canny=False)
        returns: (M_occ (B,UV,1,H,W), M_opm (B,UV,1,H,W))
        """
        b, uv, c, h, w = x.shape
        x = x.view(b * uv, c, h, w)
        if self.use_canny:
            if edge is None:
                raise ValueError('DADERP requires edge input when use_canny=True')
            edge = edge.view(b * uv, 1, h, w)
            x_in = torch.cat([x, edge], dim=1)
        else:
            x_in = x

        f0 = self.stem(x_in)                # H
        f1 = self.stage1(f0)                # H/2  (F_m1)
        f2 = self.stage2(f1)                # H/4  (F_m2)
        fh = self.stage3(f2)                # H/8  (F_h)

        # adaptive-threshold rain probability map
        m_opm = torch.sigmoid(self.opm_conv(f0) - self.tau)

        # BAEE: reverse attention on channel-wise max/avg pooled features
        max_p = f2.max(dim=1, keepdim=True)[0]
        avg_p = f2.mean(dim=1, keepdim=True)
        m_edge = 1.0 - torch.sigmoid(self.edge_conv(torch.cat([max_p, avg_p], 1)))
        f1_d = F.interpolate(f1, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        f_ee = (self.phi(f2) + self.psi(f1_d)) * m_edge
        f_ee = self.fuse_conv(f_ee)

        # DDRE: group-wise dynamic modulation
        fh_up = self.align_h(fh)
        fused = torch.cat([self.align_ee(f_ee), fh_up], dim=1)
        m_m = self.mm_conv(fused)                 # 4ch modulation maps
        f_hf = self.hf_conv(fused)                # 64ch
        g1, g2, g3, g4 = torch.chunk(f_hf, 4, dim=1)
        m1, m2, m3, m4 = torch.chunk(m_m, 4, dim=1)
        out1 = m1 * g1 + m2 * g2
        out2 = m3 * g3 + m4 * g4
        f_re = torch.cat([out1, out2], dim=1)     # 32ch

        # depth-guided filtering
        depth_d = F.interpolate(depth.view(b * uv, 1, h, w), size=f_re.shape[-2:],
                                mode='bilinear', align_corners=False)
        f_re = f_re * self.depth_gate(depth_d)

        m_occ = torch.sigmoid(self.mask_head(f_re)).view(b, uv, 1, h, w)
        m_opm = m_opm.view(b, uv, 1, h, w)
        return m_occ, m_opm
