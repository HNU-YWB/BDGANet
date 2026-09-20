import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Collate LF samples; passes through None and str fields (e.g. edge, scene)."""
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if vals[0] is None or isinstance(vals[0], str):
            out[key] = vals[0]
        else:
            out[key] = torch.stack(vals)
    return out


def sai_to_macpi(x, u=5, v=5):
    """SAI features (B, UV, C, H, W) -> Macro-Pixel Image (B, C, U*H, V*W)."""
    b, uv, c, h, w = x.shape
    x = x.view(b, u, v, c, h, w).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, u * h, v * w)


def macpi_to_sai(x, u=5, v=5):
    """Macro-Pixel Image (B, C, U*H, V*W) -> SAI features (B, UV, C, H, W)."""
    b, c, uh, vw = x.shape
    h, w = uh // u, vw // v
    x = x.view(b, c, u, h, v, w).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, u * v, c, h, w)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model, path, strict=True):
    ckpt = torch.load(path, map_location='cpu')
    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return ckpt, missing, unexpected


class _ConvBnReLU(nn.Module):
    """Conv + BN + ReLU helper shared across modules."""

    def __init__(self, c_in, c_out, k=3, s=1, p=None, act=True, bn=True):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c_out) if bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Basic residual block."""

    def __init__(self, c, stride=1):
        super().__init__()
        self.conv1 = _ConvBnReLU(c, c, 3, stride, bn=True)
        self.conv2 = _ConvBnReLU(c, c, 3, 1, act=False, bn=True)
        self.skip = nn.Sequential()
        if stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(c, c, 1, stride, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return torch.relu(self.conv2(self.conv1(x)) + self.skip(x))


class DecBlock(nn.Module):
    """Upsampling decoder block with BN-ReLU."""

    def __init__(self, c_in, c_out, scale=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        self.conv = _ConvBnReLU(c_in, c_out, 3, 1)

    def forward(self, x):
        return self.conv(self.up(x))
