"""
Depth Anything V2 wrapper (https://github.com/DepthAnything/Depth-Anything-V2).

A self-contained DPT-style implementation (timm ViT encoder + DPT head) that
loads the official pre-trained encoder weights. Used to produce the rainy-scene
depth prior D_occ and (optionally) pseudo GT depth for real scenes.

Download the official checkpoint before training, e.g.:
  python scripts/download_pretrained.py --encoder vitb
and put it under ./pretrained/ (config: depth_anything.ckpt).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class _RefineNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class DPTHead(nn.Module):
    """DPT head fusing four ViT feature scales into a depth map."""

    def __init__(self, in_channels=(96, 192, 384, 768), features=256):
        super().__init__()
        self.projects = nn.ModuleList(
            [nn.Conv2d(c, features, kernel_size=1) for c in in_channels])
        self.resize_layers = nn.ModuleList([
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False) for _ in range(4)
        ])
        self.scratch = nn.ModuleDict({
            'layer1_rn': nn.Conv2d(2 * features, features, 3, padding=1, bias=False),
            'layer2_rn': nn.Conv2d(2 * features, features, 3, padding=1, bias=False),
            'layer3_rn': nn.Conv2d(2 * features, features, 3, padding=1, bias=False),
            'layer4_rn': nn.Conv2d(2 * features, features, 3, padding=1, bias=False),
            'refinenet1': _RefineNet(features),
            'refinenet2': _RefineNet(features),
            'refinenet3': _RefineNet(features),
            'refinenet4': _RefineNet(features),
            'output_conv': nn.Sequential(
                nn.Conv2d(features, 128, 3, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(128, 32, 3, padding=1, bias=True),
                nn.ReLU(True),
                nn.Conv2d(32, 1, 1, padding=0, bias=True),
            ),
        })

    def forward(self, features):
        outs = []
        for i, x in enumerate(features):
            x = self.resize_layers[i](self.projects[i](x))
            outs.append(x)
        layer1 = outs[0] + self.scratch['layer1_rn'](torch.cat([outs[0], outs[1]], 1))
        layer2 = outs[1] + self.scratch['layer2_rn'](torch.cat([outs[1], outs[2]], 1))
        layer3 = outs[2] + self.scratch['layer3_rn'](torch.cat([outs[2], outs[3]], 1))
        layer4 = outs[3]
        path4 = self.scratch['refinenet4'](layer4)
        h3, w3 = layer3.shape[-2:]
        path3 = self.scratch['refinenet3'](layer3 + F.interpolate(path4, size=(h3, w3), mode='bilinear', align_corners=False))
        h2, w2 = layer2.shape[-2:]
        path2 = self.scratch['refinenet2'](layer2 + F.interpolate(path3, size=(h2, w2), mode='bilinear', align_corners=False))
        h1, w1 = layer1.shape[-2:]
        path1 = self.scratch['refinenet1'](layer1 + F.interpolate(path2, size=(h1, w1), mode='bilinear', align_corners=False))
        depth = self.scratch['output_conv'](path1)
        return depth


class DepthAnythingV2(nn.Module):
    """timm ViT encoder (features_only) + DPT head, loads official weights."""

    ENCODER_CFG = {
        'vits': dict(embed_dim=384, depth=12, model_name='vit_small_patch14_dinov2.lvd142m', out_ch=(96, 192, 384, 384)),
        'vitb': dict(embed_dim=768, depth=12, model_name='vit_base_patch14_dinov2.lvd142m', out_ch=(96, 192, 384, 768)),
        'vitl': dict(embed_dim=1024, depth=24, model_name='vit_large_patch14_dinov2.lvd142m', out_ch=(256, 512, 1024, 1024)),
    }

    def __init__(self, encoder='vitb', features=256, img_size=518, ckpt=None, freeze=True):
        super().__init__()
        assert encoder in self.ENCODER_CFG, encoder
        cfg = self.ENCODER_CFG[encoder]
        self.features = features
        self.img_size = img_size
        depth = cfg['depth']
        out_indices = [depth // 4 - 1, depth // 2 - 1, depth // 4 * 3 - 1, depth - 1]

        import timm
        self.backbone = timm.create_model(
            cfg['model_name'], pretrained=False, img_size=img_size,
            features_only=True, out_indices=out_indices)
        self.norm = nn.LayerNorm(cfg['embed_dim'])  # final LayerNorm of the ViT
        self.head = DPTHead(in_channels=cfg['out_ch'], features=features)
        self.layer_proj = nn.ModuleList(
            [nn.Conv2d(cfg['embed_dim'], c, 1) for c in cfg['out_ch']])

        if ckpt and os.path.exists(ckpt):
            self.load_official(ckpt)
        if freeze:
            self._freeze()

    def _freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False

    def load_official(self, ckpt):
        """Load official checkpoint (keys prefixed with 'encoder.')."""
        state = torch.load(ckpt, map_location='cpu')
        enc = {k.replace('encoder.', ''): v for k, v in state.items() if k.startswith('encoder.')}
        missing, unexpected = self.load_state_dict(enc, strict=False)
        if missing:
            print(f'[DepthAnythingV2] missing keys: {missing[:5]} ... (head part, fine)')

    def forward(self, x):
        """x: (B, 3, H, W) in [0,1] -> (B, 1, H, W) raw relative depth."""
        b, c, h, w = x.shape
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        feats = self.backbone(x)                      # list of (B, D, 37, 37)
        # apply final LayerNorm on the deepest scale (token layout)
        f = feats[-1]
        b2, d2, h2, w2 = f.shape
        feats[-1] = self.norm(f.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b2, d2, h2, w2)
        proj = [self.layer_proj[i](f) for i, f in enumerate(feats)]
        depth = self.head(proj)
        depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)
        return depth


def build_depth_estimator(cfg, device):
    """Build and move the depth estimator; returns None if not configured."""
    dcfg = cfg.get('depth_anything', {})
    if not dcfg.get('ckpt'):
        return None
    model = DepthAnythingV2(encoder=dcfg.get('encoder', 'vitb'),
                            img_size=dcfg.get('img_size', 518),
                            ckpt=dcfg['ckpt'],
                            freeze=dcfg.get('freeze', True))
    model.eval()
    return model.to(device)


@torch.no_grad()
def estimate_depth(depth_model, lf, center_view=12, all_views=True):
    """
    Estimate per-view depth for a rainy LF.
    lf: (B, UV, 3, H, W) in [0,1] -> (B, UV, 1, H, W) normalized to [0,1].
    If all_views=False, estimates only the center view and broadcasts it.
    """
    b, uv, c, h, w = lf.shape
    views = list(range(uv)) if all_views else [center_view]
    depths = []
    for v in views:
        d = depth_model(lf[:, v])  # (B, 1, H, W)
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        depths.append(d)
    if not all_views:
        d = depths[0].unsqueeze(1).repeat(1, uv, 1, 1, 1)
        return d
    return torch.stack(depths, dim=1)
