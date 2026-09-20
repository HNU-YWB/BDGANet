"""Train BDGANet.

Usage:
  python tools/train.py --config configs/bdganet_rlfdb.yaml
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from datasets import LFDataset
from models import BDGANet, Losses
from utils.misc import set_seed, save_checkpoint, load_checkpoint, collate_fn
from utils.metrics import calculate_psnr_ssim_torch
from utils.depth_anything_v2 import build_depth_estimator, estimate_depth


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/bdganet_rlfdb.yaml')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, cfg, device, depth_estimator=None):
    model.eval()
    psnrs, ssims = [], []
    for batch in loader:
        lf = batch['lf'].to(device)
        gt = batch['gt']
        if gt is None:
            continue
        gt = gt.to(device)
        d_occ = batch['d_occ']
        if d_occ is None and depth_estimator is not None:
            d_occ = estimate_depth(depth_estimator, lf, cfg['model']['center_view'])
        d_occ = d_occ.to(device)
        edge = batch['edge'].to(device) if batch['edge'] is not None else None
        outs = model(lf, d_occ, edge=edge, mode='infer')
        p_list, s_list = calculate_psnr_ssim_torch(outs['i_out'], gt, border=cfg['test'].get('border', 0))
        psnrs += p_list
        ssims += s_list
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('seed', 42))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    save_dir = cfg['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))

    train_ds = LFDataset(cfg, split=cfg['data']['train_split'], train=True)
    val_ds = LFDataset(cfg, split=cfg['data']['test_split'], train=False)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['data']['num_workers'], pin_memory=pin,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg['data']['num_workers'], pin_memory=pin,
                            collate_fn=collate_fn)

    model = BDGANet(
        in_ch=cfg['model']['in_ch'], base_ch=cfg['model']['base_ch'],
        mask_ch=cfg['model']['mask_ch'], depth_ch=cfg['model']['depth_ch'],
        bottleneck=cfg['model']['bottleneck_ch'],
        angular_u=cfg['model']['angular_u'], angular_v=cfg['model']['angular_v'],
        center_view=cfg['model']['center_view'],
        use_canny=cfg['model'].get('use_canny', False)).to(device)

    depth_estimator = build_depth_estimator(cfg, device)
    losses = Losses(cfg, center_view=cfg['model']['center_view']).to(device)

    optimizer = Adam([
        {'params': model.parameters()},
        {'params': [p for p in losses.parameters() if p.requires_grad], 'lr': cfg['train']['lr']},
    ], lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    epochs = cfg['train']['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg['train']['lr_min'])

    start_epoch = 0
    if cfg['train'].get('resume'):
        ckpt, _, _ = load_checkpoint(model, cfg['train']['resume'], strict=False)
        start_epoch = ckpt.get('epoch', 0)

    best_psnr = 0.0
    print(f'[BDGANet] train on {len(train_ds)} scenes, validate on {len(val_ds)} scenes')
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            lf = batch['lf'].to(device)
            gt = batch['gt'].to(device)
            d_gt = batch['d_gt'].to(device)
            d_occ = batch['d_occ']
            if d_occ is None:
                with torch.no_grad():
                    d_occ = estimate_depth(depth_estimator, lf, cfg['model']['center_view'])
            else:
                d_occ = d_occ.to(device)
            edge = batch['edge'].to(device) if batch['edge'] is not None else None

            outs = model(lf, d_occ, edge=edge, i_gt=gt, d_gt=d_gt, mode='train')
            loss, items = losses(outs, lf, gt, d_gt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            optimizer.step()

            if it % cfg['train']['log_freq'] == 0:
                msg = (f"[{epoch:3d}/{epochs}] it {it:4d} loss {loss.item():.4f} "
                       f"rec {items['L_rec'].item():.4f} p {items['L_p'].item():.4f} "
                       f"cr {items['L_cr'].item():.4f} dd {items['L_dd'].item():.4f} "
                       f"ap {items['L_ap'].item():.4f}")
                print(msg)
                for k, v in items.items():
                    writer.add_scalar(f'train/{k}', v.item(), epoch * len(train_loader) + it)

        scheduler.step()
        print(f'[epoch {epoch}] done in {time.time() - t0:.1f}s, lr={optimizer.param_groups[0]["lr"]:.2e}')

        if (epoch + 1) % cfg['train']['eval_every'] == 0 or epoch == epochs - 1:
            psnr, ssim = validate(model, val_loader, cfg, device, depth_estimator)
            print(f'[eval] epoch {epoch}: PSNR {psnr:.3f} dB, SSIM {ssim:.4f}')
            writer.add_scalar('val/psnr', psnr, epoch)
            writer.add_scalar('val/ssim', ssim, epoch)
            if psnr > best_psnr:
                best_psnr = psnr
                save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                 'best_psnr': best_psnr},
                                os.path.join(save_dir, 'best.pth'))
        if (epoch + 1) % 10 == 0:
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                            os.path.join(save_dir, f'epoch_{epoch + 1}.pth'))

    save_checkpoint({'epoch': epochs, 'state_dict': model.state_dict()},
                    os.path.join(save_dir, 'final.pth'))
    print(f'[done] best PSNR {best_psnr:.3f} dB, ckpts saved to {save_dir}')


if __name__ == '__main__':
    main()
