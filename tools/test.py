"""Evaluate BDGANet on a test split, save restored images and report PSNR/SSIM.

Usage:
  python tools/test.py --config configs/bdganet_rlfdb.yaml --ckpt experiments/BDGANet_RLFDB/best.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from PIL import Image

from datasets import LFDataset
from models import BDGANet
from utils.misc import set_seed, load_checkpoint, collate_fn
from utils.metrics import calculate_psnr_ssim_torch
from utils.depth_anything_v2 import build_depth_estimator, estimate_depth


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/bdganet_rlfdb.yaml')
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def save_image(tensor, path):
    img = tensor.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).round().astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('seed', 42))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = cfg['test']['save_results']
    os.makedirs(out_dir, exist_ok=True)

    test_ds = LFDataset(cfg, split=cfg['data']['test_split'], train=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=cfg['data']['num_workers'], collate_fn=collate_fn)

    model = BDGANet(
        in_ch=cfg['model']['in_ch'], base_ch=cfg['model']['base_ch'],
        mask_ch=cfg['model']['mask_ch'], depth_ch=cfg['model']['depth_ch'],
        bottleneck=cfg['model']['bottleneck_ch'],
        angular_u=cfg['model']['angular_u'], angular_v=cfg['model']['angular_v'],
        center_view=cfg['model']['center_view'],
        use_canny=cfg['model'].get('use_canny', False)).to(device)
    ckpt, _, _ = load_checkpoint(model, args.ckpt)
    model.eval()

    depth_estimator = build_depth_estimator(cfg, device)
    psnrs, ssims, names = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            lf = batch['lf'].to(device)
            gt = batch['gt']
            d_occ = batch['d_occ']
            if d_occ is None and depth_estimator is not None:
                d_occ = estimate_depth(depth_estimator, lf, cfg['model']['center_view'])
            if d_occ is None:
                raise RuntimeError('no depth prior available; run tools/gen_depth.py first')
            d_occ = d_occ.to(device)
            edge = batch['edge'].to(device) if batch['edge'] is not None else None

            outs = model(lf, d_occ, edge=edge, mode='infer')
            scene = batch['scene'][0]
            save_image(outs['i_out'][0], os.path.join(out_dir, f'{scene}_derained.png'))
            save_image(lf[0, cfg['model']['center_view']], os.path.join(out_dir, f'{scene}_rainy.png'))
            if gt is not None:
                gt = gt.to(device)
                p_list, s_list = calculate_psnr_ssim_torch(
                    outs['i_out'], gt, border=cfg['test'].get('border', 0))
                psnrs += p_list
                ssims += s_list
                names.append(scene)
                print(f'{scene}: PSNR {p_list[0]:.2f} dB, SSIM {s_list[0]:.4f}')

    if psnrs:
        print(f'\nAverage PSNR: {np.mean(psnrs):.3f} dB, SSIM: {np.mean(ssims):.4f} '
              f'(over {len(psnrs)} scenes)')
    print(f'Results saved to {out_dir}')


if __name__ == '__main__':
    main()
