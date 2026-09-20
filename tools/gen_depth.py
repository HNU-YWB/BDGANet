"""Precompute per-view depth priors (Depth Anything V2) for all scenes.

Writes depth_est.npy (U*V, 1, H, W, normalized to [0,1]) into each scene dir.
Optionally writes depth_gt.npy = depth_est of the center view when no GT depth
is available, to be used as pseudo GT during training.

Usage:
  python tools/gen_depth.py --config configs/bdganet_rlfdb.yaml --split train
  python tools/gen_depth.py --config configs/bdganet_rlfdb.yaml --split test
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets.lf_dataset import LFDataset
from utils.depth_anything_v2 import build_depth_estimator, estimate_depth


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/bdganet_rlfdb.yaml')
    p.add_argument('--split', type=str, required=True, choices=['train', 'test'])
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--write_pseudo_gt', action='store_true',
                   help='write depth_gt.npy from center-view estimation')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ds = LFDataset(cfg, split=args.split, train=False)
    est = build_depth_estimator(cfg, device)
    assert est is not None, 'depth_anything.ckpt not configured; download the weights first'

    center = cfg['model']['center_view']
    for idx in tqdm(range(len(ds)), desc=f'gen depth [{args.split}]'):
        sample = ds[idx]
        lf = sample['lf'].unsqueeze(0).to(device)
        scene = sample['scene']
        scene_dir = os.path.join(cfg['data']['root'], args.split, scene)
        with torch.no_grad():
            d = estimate_depth(est, lf, center_view=center, all_views=True)
        d = d[0].cpu().numpy().astype(np.float32)  # (UV, 1, H, W)
        np.save(os.path.join(scene_dir, 'depth_est.npy'), d)
        if args.write_pseudo_gt:
            np.save(os.path.join(scene_dir, 'depth_gt.npy'), d[center])
    print('done')


if __name__ == '__main__':
    main()
