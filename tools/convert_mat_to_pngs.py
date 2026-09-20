"""Convert LF rainy .mat files (RLFDB / RLMB style) into per-scene PNG folders.

Adjust LF_KEY / GT_KEY / DEPTH_KEY to the variable names used by your dataset.

Expected .mat structure (common variants):
  - LF  : (U, V, H, W, C) uint8/float rainy light field, or (U*V, H, W, C)
  - GT  : (H, W, C) clean center view (may be absent for real scenes)
  - Dis : (U, V, H, W) disparity / depth (optional)

Usage:
  python tools/convert_mat_to_pngs.py --src /path/to/mats --dst ./data/RLFDB --split train
"""

import argparse
import os
import glob
import numpy as np
import scipy.io as sio
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=str, required=True, help='dir containing .mat files')
    p.add_argument('--dst', type=str, required=True, help='dataset root (e.g. ./data/RLFDB)')
    p.add_argument('--split', type=str, default='train')
    p.add_argument('--lf_key', type=str, default='LF')
    p.add_argument('--gt_key', type=str, default='GT')
    p.add_argument('--depth_key', type=str, default='Dis')
    p.add_argument('--angular', type=int, default=5)
    return p.parse_args()


def to_uint8(x):
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255).round().astype(np.uint8)
    return x


def main():
    args = parse_args()
    mats = sorted(glob.glob(os.path.join(args.src, '*.mat')))
    assert mats, f'no .mat files under {args.src}'
    os.makedirs(args.dst, exist_ok=True)
    for i, mpath in enumerate(mats):
        data = sio.loadmat(mpath)
        lf = data[args.lf_key]
        if lf.ndim == 4 and lf.shape[0] == args.angular ** 2:
            uv, h, w, c = lf.shape          # (U*V, H, W, C)
        elif lf.ndim == 5:
            u, v, h, w, c = lf.shape        # (U, V, H, W, C)
            lf = lf.reshape(-1, h, w, c)
        else:
            raise ValueError(f'unexpected LF shape {lf.shape}')
        scene_dir = os.path.join(args.dst, args.split, f'scene_{i:04d}')
        os.makedirs(scene_dir, exist_ok=True)
        for v in range(lf.shape[0]):
            Image.fromarray(to_uint8(lf[v])).save(os.path.join(scene_dir, f'view_{v:02d}.png'))
        if args.gt_key in data and data[args.gt_key].size > 1:
            gt = data[args.gt_key]
            if gt.ndim == 5:
                gt = gt[gt.shape[0] // 2, gt.shape[1] // 2]
            elif gt.ndim == 4 and gt.shape[0] == args.angular ** 2:
                gt = gt[args.angular ** 2 // 2]
            Image.fromarray(to_uint8(gt)).save(os.path.join(scene_dir, 'gt_center.png'))
        if args.depth_key in data and data[args.depth_key].size > 1:
            dis = data[args.depth_key]
            if dis.ndim >= 3:
                dis = dis[dis.shape[0] // 2, dis.shape[1] // 2] if dis.ndim == 4 else dis[dis.shape[0] // 2]
            np.save(os.path.join(scene_dir, 'depth_gt.npy'), dis.astype(np.float32))
        print(f'[{i + 1}/{len(mats)}] {os.path.basename(mpath)} -> {scene_dir}')


if __name__ == '__main__':
    main()
