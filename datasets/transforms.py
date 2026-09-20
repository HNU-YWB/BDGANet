import numpy as np
import random


class RandomCrop:
    """Randomly crop LF volumes (views, gt, depths, edges) to patch_size."""

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        h, w = sample['lf'].shape[-2:]
        ph, pw = self.patch_size, self.patch_size
        if h == ph and w == pw:
            return sample
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
        for key in ('lf', 'gt', 'd_occ', 'd_gt', 'edge', 'depth_est'):
            if key in sample and sample[key] is not None:
                sample[key] = sample[key][..., top:top + ph, left:left + pw]
        return sample


class RandomFlip:
    def __call__(self, sample):
        if random.random() < 0.5:
            for key in ('lf', 'gt', 'd_occ', 'd_gt', 'edge', 'depth_est'):
                if key in sample and sample[key] is not None:
                    sample[key] = np.flip(sample[key], axis=-1)
        if random.random() < 0.5:
            for key in ('lf', 'gt', 'd_occ', 'd_gt', 'edge', 'depth_est'):
                if key in sample and sample[key] is not None:
                    sample[key] = np.flip(sample[key], axis=-2)
        return sample


class RandomRot90:
    def __call__(self, sample):
        k = random.randint(0, 3)
        if k == 0:
            return sample
        for key in ('lf', 'gt', 'd_occ', 'd_gt', 'edge', 'depth_est'):
            if key in sample and sample[key] is not None:
                sample[key] = np.rot90(sample[key], k, axes=(-2, -1))
        return sample


class ToTensor:
    """Convert float arrays (0-1) to float32 torch tensors."""

    def __call__(self, sample):
        import torch
        out = {}
        for key, val in sample.items():
            if val is None or isinstance(val, str):
                out[key] = val
            else:
                out[key] = torch.from_numpy(np.ascontiguousarray(val, dtype=np.float32))
        return out
