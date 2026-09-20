import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import RandomCrop, RandomFlip, RandomRot90, ToTensor


def _read_image(path):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # (C, H, W)


def _read_depth(path):
    """Read depth npy (H, W) or (UV, 1, H, W) / single-channel image, normalize to [0,1]."""
    if path is None or not os.path.exists(path):
        return None
    if path.endswith('.npy'):
        d = np.load(path)
    else:
        d = np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    d = d.astype(np.float32)
    if d.ndim == 2:
        d = d[None, ...]  # (1, H, W)
    mn, mx = d.min(), d.max()
    if mx - mn > 1e-6:
        d = (d - mn) / (mx - mn)
    return d


class LFDataset(Dataset):
    """
    Light field rainy dataset.

    Directory layout (see README):
      root/train/scene_0001/view_00.png ... view_24.png   # 5x5 SAIs, center = view_12
      root/train/scene_0001/gt_center.png                 # clean center view (test: optional)
      root/train/scene_0001/depth_gt.npy                  # optional GT depth of center view
      root/train/scene_0001/depth_est.npy                 # precomputed depth (U*V, 1, H, W)

    Sample fields:
      lf      (UV, 3, H, W) rainy sub-aperture images
      gt      (3, H, W)    clean center view (None if absent)
      d_occ   (UV, 1, H, W) rainy-scene depth prior from Depth Anything
      d_gt    (1, H, W)    GT depth (fallback: depth_est of center view)
      edge    (UV, 1, H, W) Canny edges (optional)
    """

    def __init__(self, cfg, split='train', train=True):
        self.cfg = cfg
        self.root = os.path.join(cfg['data']['root'], split)
        self.train = train
        self.uv = cfg['model']['angular_u'] * cfg['model']['angular_v']
        self.center = cfg['model']['center_view']
        self.scenes = sorted(glob.glob(os.path.join(self.root, '*')))
        assert len(self.scenes) > 0, f'no scenes found under {self.root}'
        self.use_gt_depth = cfg['data'].get('use_gt_depth', True)
        self.precompute_depth = cfg['data'].get('precompute_depth', True)
        self.canny = None
        if cfg['model'].get('use_canny', False):
            from utils.canny import canny_edges
            self.canny = canny_edges

    def __len__(self):
        return len(self.scenes)

    def _load_views(self, scene):
        views = []
        for v in range(self.uv):
            p = os.path.join(scene, f'view_{v:02d}.png')
            views.append(_read_image(p))
        return np.stack(views, axis=0)  # (UV, 3, H, W)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        lf = self._load_views(scene)
        _, _, h, w = lf.shape

        gt = _read_image(os.path.join(scene, 'gt_center.png')) if os.path.exists(
            os.path.join(scene, 'gt_center.png')) else None

        # rainy-scene depth prior (Depth Anything), per view
        d_occ = None
        if self.precompute_depth:
            d_occ = _read_depth(os.path.join(scene, 'depth_est.npy'))
            if d_occ is not None and d_occ.shape[0] == 1:
                d_occ = np.repeat(d_occ, self.uv, axis=0)

        # GT depth for training (center view)
        d_gt = None
        if self.train:
            if self.use_gt_depth:
                d_gt = _read_depth(os.path.join(scene, 'depth_gt.npy'))
            if d_gt is None and self.precompute_depth and d_occ is not None:
                d_gt = d_occ[self.center]

        edge = None
        if self.canny is not None:
            edges = []
            for v in range(self.uv):
                e = self.canny(lf[v].transpose(1, 2, 0))  # (H, W)
                edges.append(e[None, ...])
            edge = np.stack(edges, axis=0)

        sample = {'lf': lf, 'gt': gt, 'd_occ': d_occ, 'd_gt': d_gt, 'edge': edge,
                  'scene': os.path.basename(scene)}
        if self.train:
            sample = RandomCrop(self.cfg['data']['patch_size'])(sample)
            sample = RandomFlip()(sample)
            sample = RandomRot90()(sample)
        return ToTensor()(sample)
