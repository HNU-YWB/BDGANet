"""Loss functions of BDGANet.

Appearance branch: L_arec = L1 + lambda_2 * L_p + L_cr
Geometry branch:   L_degeo = L_dd + (L_lap + lambda_1 * L_reg) + L_tv + L_dp + L_ap
Total:             L = L_deapp + L_degeo,  L_deapp = L_arec + L_dp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


_VGG19_RELU_NAMES = ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2',
                     'relu3_3', 'relu3_4', 'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4',
                     'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4']


class _VGGFeatures(nn.Module):
    """Multi-scale VGG19 features for perceptual / contrastive losses."""

    def __init__(self, layer_names=('relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4')):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features
        children = list(vgg.children())
        self.layers = nn.ModuleDict()
        cur_conv = 0
        for idx, m in enumerate(children):
            if isinstance(m, nn.Conv2d):
                cur_conv += 1
            elif isinstance(m, nn.ReLU) and cur_conv > 0:
                name = _VGG19_RELU_NAMES[cur_conv - 1]
                if name in layer_names:
                    self.layers[name] = nn.Sequential(*children[:idx + 1])
                if len(self.layers) == len(layer_names):
                    break
        assert len(self.layers) == len(layer_names), 'VGG feature layers not found'
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return [m(x) for m in self.layers.values()]


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = _VGGFeatures()

    def forward(self, pred, gt):
        loss = 0.0
        for fp, fg in zip(self.vgg(pred), self.vgg(gt)):
            loss += F.l1_loss(fp, fg)
        return loss


class ContrastiveLoss(nn.Module):
    """L_cr: pulls I_out close to I_gt, pushes it away from the rainy reference."""

    def __init__(self, weights=(1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0), eps=1e-7):
        super().__init__()
        self.weights = weights
        self.eps = eps
        self.vgg = _VGGFeatures()
        # learnable projection for the deepest semantic layer
        self.proj = nn.Conv2d(512, 512, 1)

    def forward(self, out, gt, rainy):
        f_out = self.vgg(out)
        f_gt = self.vgg(gt)
        f_rain = self.vgg(rainy)
        loss = 0.0
        for i, (fo, fg, fr) in enumerate(zip(f_out, f_gt, f_rain)):
            if i == len(f_out) - 1:
                fo, fg, fr = self.proj(fo), self.proj(fg), self.proj(fr)
            num = (fo - fg).abs().sum()
            den = (fo - fr).abs().sum() + self.eps
            loss += self.weights[i] * num / den
        return loss


class LaplacianLoss(nn.Module):
    """Second-order geometric consistency between composite depth and GT depth."""

    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3))

    def forward(self, pred, gt):
        lp = F.conv2d(pred, self.kernel, padding=1)
        lg = F.conv2d(gt, self.kernel, padding=1)
        return ((lp - lg) ** 2).mean()


class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        dx = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
        dy = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        return dx + dy


class Losses(nn.Module):
    """Aggregates all BDGANet losses; see class docstring for the formula."""

    def __init__(self, cfg, center_view=12):
        super().__init__()
        lw = cfg['loss']
        self.lambda1 = lw['reg']          # 0.1, inside L_ldr
        self.lambda2 = lw['perceptual']   # 0.5, weight of L_p
        self.w = {k: lw.get(k, 1.0) for k in ('rec', 'contrastive', 'dd', 'ldr', 'tv', 'dp', 'ap')}
        self.center = center_view
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()
        self.contrastive = ContrastiveLoss()
        self.laplacian = LaplacianLoss()
        self.tv = TVLoss()

    def forward(self, outs, lf, gt, d_gt):
        """
        outs: output dict of BDGANet (train mode)
        lf:   (B, UV, 3, H, W)
        gt:   (B, 3, H, W) clean center view
        d_gt: (B, 1, H, W) GT depth of center view
        """
        i_out = outs['i_out']
        d_de_c = outs['d_de_c']
        d_de_r = outs['d_de_r']
        m_occ_c = outs['m_occ'][:, self.center]
        w_occ = outs['w_occ']
        rainy_c = lf[:, self.center]

        # ---- appearance branch ----
        l_rec = self.l1(i_out, gt)
        l_p = self.perceptual(i_out, gt)
        l_cr = self.contrastive(i_out, gt, rainy_c)
        l_dp = self.l1(d_de_r, d_gt)
        l_deapp = self.w['rec'] * l_rec + self.lambda2 * l_p + self.w['contrastive'] * l_cr + self.w['dp'] * l_dp

        # ---- geometry branch ----
        d_hat = m_occ_c * d_de_c + (1 - m_occ_c) * d_gt
        l_lap = self.laplacian(d_hat, d_gt)
        l_reg = d_hat.abs().mean()
        l_ldr = l_lap + self.lambda1 * l_reg
        l_tv = self.tv(d_de_c)
        l_dd = self.l1(d_de_c, d_gt)
        l_ap = (w_occ * (d_de_r - d_gt)).abs().mean()
        l_degeo = self.w['dd'] * l_dd + self.w['ldr'] * l_ldr + self.w['tv'] * l_tv \
            + self.w['dp'] * l_dp + self.w['ap'] * l_ap

        total = l_deapp + l_degeo
        items = {'L_total': total, 'L_rec': l_rec, 'L_p': l_p, 'L_cr': l_cr,
                 'L_dp': l_dp, 'L_deapp': l_deapp, 'L_dd': l_dd, 'L_lap': l_lap,
                 'L_reg': l_reg, 'L_ldr': l_ldr, 'L_tv': l_tv, 'L_ap': l_ap,
                 'L_degeo': l_degeo}
        return total, items
