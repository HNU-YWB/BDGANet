"""Report model parameters and FLOPs (for Table 3 reproduction).

Usage:
  python tools/benchmark.py --config configs/bdganet_rlfdb.yaml --size 128
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import yaml
import torch
from thop import profile

from models import BDGANet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/bdganet_rlfdb.yaml')
    p.add_argument('--size', type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    m = cfg['model']
    model = BDGANet(in_ch=m['in_ch'], base_ch=m['base_ch'], mask_ch=m['mask_ch'],
                    depth_ch=m['depth_ch'], bottleneck=m['bottleneck_ch'],
                    angular_u=m['angular_u'], angular_v=m['angular_v'],
                    center_view=m['center_view'], use_canny=m.get('use_canny', False))
    model.eval()
    uv = m['angular_u'] * m['angular_v']
    lf = torch.randn(1, uv, 3, args.size, args.size)
    d_occ = torch.rand(1, uv, 1, args.size, args.size)
    flops, params = profile(model, inputs=(lf, d_occ), verbose=False)
    print(f'Params: {params / 1e6:.2f} M')
    print(f'FLOPs:  {flops / 1e9:.2f} G')


if __name__ == '__main__':
    main()
