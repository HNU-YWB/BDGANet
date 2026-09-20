"""Adaptive View Aggregation Attention (AVAA).

Projects reference-view and auxiliary-view features into a low-dimensional
bottleneck space, computes per-position angular relevance between the reference
view and every auxiliary view, and adaptively aggregates informative auxiliary
features while suppressing rain-corrupted / redundant ones.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import _ConvBnReLU


class AVAA(nn.Module):
    def __init__(self, in_ch=64, bottleneck=16, downsample=8):
        super().__init__()
        self.bottleneck = bottleneck
        self.downsample = downsample
        self.w_q = nn.Conv2d(in_ch, bottleneck, 1)
        self.w_k = nn.Conv2d(in_ch, bottleneck, 1)
        self.w_v = nn.Conv2d(in_ch, bottleneck, 1)
        self.proj_out = nn.Conv2d(bottleneck, in_ch, 1)
        self.fuse = nn.Sequential(
            _ConvBnReLU(in_ch * 2, in_ch, 3, 1),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1))

    def forward(self, f_sai, center_idx=12):
        """
        f_sai: (B, UV, C, H, W) SAI features from MacPI
        returns: (B, C, H, W) enhanced reference-view features
        """
        b, uv, c, h, w = f_sai.shape
        f_r = f_sai[:, center_idx]                     # (B, C, H, W)
        idx = [i for i in range(uv) if i != center_idx]
        n = len(idx)
        f_aux = f_sai[:, idx]                          # (B, N, C, H, W)

        # low-dimensional bottleneck + spatial downsampling
        hp, wp = h // self.downsample, w // self.downsample
        q = F.adaptive_avg_pool2d(self.w_q(f_r), (hp, wp))
        k = F.adaptive_avg_pool2d(
            self.w_k(f_aux.reshape(b * n, c, h, w)), (hp, wp)).view(b, n, self.bottleneck, hp, wp)
        v = F.adaptive_avg_pool2d(
            self.w_v(f_aux.reshape(b * n, c, h, w)), (hp, wp)).view(b, n, self.bottleneck, hp, wp)

        # per-position angular attention: scores over auxiliary views
        scores = (q.unsqueeze(1) * k).sum(dim=2) / math.sqrt(self.bottleneck)  # (B,N,h',w')
        attn = torch.softmax(scores, dim=1)
        out_low = (attn.unsqueeze(2) * v).sum(dim=1)   # (B, bottleneck, h', w')

        out_up = F.interpolate(out_low, size=(h, w), mode='bilinear', align_corners=False)
        m_attn = torch.sigmoid(self.proj_out(out_up))  # importance of auxiliary info

        # adaptive aggregation into the reference view
        agg = (f_aux * m_attn.unsqueeze(1)).mean(dim=1)
        f_out = f_r + self.fuse(torch.cat([f_r, agg], dim=1))
        return f_out
