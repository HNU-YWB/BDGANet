"""BDGANet: Bidirectional Difference-Perception Geometry-Appearance Interaction
Network for Light Field Deraining.

Pipeline: rainy LF -> (Depth Anything prior) -> DA-RERP rain mask -> GAID
(geometry + appearance + bidirectional difference perception) -> clean center view.
"""

import torch
import torch.nn as nn

from .daderp import DADERP
from .gaid import GAID


class BDGANet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, mask_ch=64, depth_ch=32, bottleneck=16,
                 angular_u=5, angular_v=5, center_view=12, use_canny=False):
        super().__init__()
        self.uv = angular_u * angular_v
        self.center = center_view
        self.daderp = DADERP(in_ch=in_ch, base_ch=mask_ch, use_canny=use_canny)
        self.gaid = GAID(in_ch=in_ch, base_ch=base_ch, angular_u=angular_u,
                         angular_v=angular_v, center_view=center_view,
                         depth_ch=depth_ch, bottleneck=bottleneck)

    def forward(self, lf, d_occ, edge=None, i_gt=None, d_gt=None, mode='infer'):
        """
        lf:    (B, UV, 3, H, W) rainy SAI volume in [0,1]
        d_occ: (B, UV, 1, H, W) rainy-scene depth prior in [0,1]
        edge:  (B, UV, 1, H, W) optional Canny edges
        mode:  'infer' or 'train' (train needs i_gt / d_gt of the center view)

        returns dict:
          i_out  (B,3,H,W) clean center view
          d_de   (B,UV,1,H,W) rain-free depth of all views
          d_de_c (B,1,H,W) rain-free depth of center view
          m_occ  (B,UV,1,H,W) rain mask
          (train) d_de_r, w_occ
        """
        m_occ, m_opm = self.daderp(lf, d_occ, edge)
        outs = self.gaid(lf, d_occ, m_occ, i_gt_c=i_gt, d_gt_c=d_gt, mode=mode)
        outs['m_occ'] = m_occ
        outs['m_opm'] = m_opm
        return outs
