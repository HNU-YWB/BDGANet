import numpy as np
import torch
import torch.nn.functional as F


def calculate_psnr(img1, img2, border=0, data_range=1.0):
    """PSNR on numpy arrays (H, W, C) or (C, H, W), values in [0, 1]."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    if img1.ndim == 3 and img1.shape[0] in (1, 3):
        img1 = img1.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)
    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(data_range * data_range / mse)


def _fspecial_gauss(size, sigma):
    coords = np.arange(size, dtype=np.float32) - size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def calculate_ssim(img1, img2, border=0, data_range=1.0, win_size=11, sigma=1.5):
    """SSIM on numpy arrays, values in [0, 1]."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    if img1.ndim == 3 and img1.shape[0] in (1, 3):
        img1 = img1.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)
    if img1.ndim == 2:
        img1 = img1[..., None]
        img2 = img2[..., None]
    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]
    img1 = img1.astype(np.float64) / data_range
    img2 = img2.astype(np.float64) / data_range
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    pad = win_size // 2
    kernel_1d = _fspecial_gauss(win_size, sigma)
    window = np.outer(kernel_1d, kernel_1d)[..., None]  # (W, W, 1)
    window = np.repeat(window, img1.shape[2], axis=2)

    def conv2d(x):
        x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        return _conv2d_numpy(x, window)

    mu1, mu2 = conv2d(img1), conv2d(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = conv2d(img1 ** 2) - mu1_sq
    sigma2_sq = conv2d(img2 ** 2) - mu2_sq
    sigma12 = conv2d(img1 * img2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def _conv2d_numpy(x, window):
    """Valid 2D convolution per channel with a shared 2D kernel."""
    h, w, c = x.shape
    kw, kh = window.shape[:2]
    out_h, out_w = h - kh + 1, w - kw + 1
    out = np.zeros((out_h, out_w, c), dtype=np.float64)
    for i in range(c):
        out[..., i] = _conv2d_single(x[..., i], window[..., 0])
    return out


def _conv2d_single(x, w):
    kh, kw = w.shape
    oh, ow = x.shape[0] - kh + 1, x.shape[1] - kw + 1
    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += w[i, j] * x[i:i + oh, j:j + ow]
    return out


def calculate_psnr_ssim_torch(pred, gt, border=0, data_range=1.0):
    """Batch PSNR/SSIM for tensors (B, C, H, W) in [0,1]."""
    pred = pred.clamp(0, 1).detach().float()
    gt = gt.clamp(0, 1).detach().float()
    b = pred.shape[0]
    psnr_list, ssim_list = [], []
    for i in range(b):
        p, g = pred[i], gt[i]
        if border > 0:
            p = p[..., border:-border, border:-border]
            g = g[..., border:-border, border:-border]
        mse = F.mse_loss(p, g).item()
        psnr_list.append(10 * np.log10(data_range ** 2 / (mse + 1e-12)))
        ssim_list.append(_ssim_torch(p, g, data_range))
    return psnr_list, ssim_list


def _ssim_torch(img1, img2, data_range=1.0, win_size=11, sigma=1.5):
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    g = torch.arange(win_size, dtype=img1.dtype, device=img1.device).float() - win_size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    window = window.expand(img1.shape[0], 1, win_size, win_size).contiguous()
    pad = win_size // 2
    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=pad, groups=img1.shape[0])
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=pad, groups=img2.shape[0])
    mu1 = mu1.squeeze(0)
    mu2 = mu2.squeeze(0)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d((img1 ** 2).unsqueeze(0), window, padding=pad, groups=img1.shape[0]).squeeze(0) - mu1_sq
    sigma2_sq = F.conv2d((img2 ** 2).unsqueeze(0), window, padding=pad, groups=img2.shape[0]).squeeze(0) - mu2_sq
    sigma12 = F.conv2d((img1 * img2).unsqueeze(0), window, padding=pad, groups=img1.shape[0]).squeeze(0) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())
