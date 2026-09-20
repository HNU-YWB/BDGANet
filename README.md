# BDGANet: Bidirectional Difference-Perception Geometry-Appearance
# Interaction Network for Light Field Deraining

Official implementation of the paper **"BDGANet: Bidirectional Difference-Perception Network for Light Field Deraining with Geometry-Appearance Interaction"** (submitted to Neurocomputing).

BDGANet jointly restores rain-free geometry (depth) and clean appearance for 4D light fields under rain degradation:

- **DA-RERP** — Depth-Assisted Rain Edge-Region Predictor: accurate rain-mask localization via adaptive-threshold rain prediction, boundary-aware edge enhancement (reverse attention) and depth-guided dynamic region enhancement.
- **GAID** — Geometry-Appearance Interactive Derainer: a geometry reconstructor (RADepth-style) restores rain-free depth; a depth-guided appearance reconstructor (DGSE + AVAA) restores the clean center view.
- **Bidirectional Difference-Perception** — geometry-difference perception feeds appearance errors back to depth refinement, while appearance-difference perception reweights the geometry objective toward regions with unsatisfactory deraining.
- **AVAA** — Adaptive View Aggregation Attention: selectively aggregates informative sub-aperture views and suppresses rain-corrupted / redundant ones.

## 1. Environment

- Python >= 3.8, PyTorch >= 1.12, CUDA 11.6+ (tested on a single NVIDIA RTX 4090)
- Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Data preparation

### 2.1 Dataset layout

Organize RLFDB / RLMB under `data/<DATASET>/` as follows (5×5 angular grid, center view index = 12):

```
data/RLFDB/
├── train/
│   └── scene_0001/
│       ├── view_00.png ~ view_24.png   # rainy sub-aperture images
│       ├── gt_center.png               # clean center view (optional for real scenes)
│       ├── depth_gt.npy                # optional GT depth of center view (H, W)
│       └── depth_est.npy               # precomputed per-view depth prior (U*V, 1, H, W)
└── test/
    └── ...
```

- If your dataset ships as `.mat` files, convert them first:

```bash
python tools/convert_mat_to_pngs.py --src /path/to/mats --dst ./data/RLFDB --split train
python tools/convert_mat_to_pngs.py --src /path/to/mats --dst ./data/RLFDB --split test
```

  Adjust `--lf_key / --gt_key / --depth_key` to the variable names of your `.mat` files.

- For real-world scenes without GT, simply omit `gt_center.png`; the test script then only saves restored images.

### 2.2 Depth priors

The rainy-scene depth prior `D_occ` is produced by [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (frozen during training).

1. Download the checkpoint:

```bash
python scripts/download_pretrained.py --encoder vitb
# or: wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
# put it at ./pretrained/depth_anything_v2_vitb.pth
```

2. Precompute per-view depth for every scene (recommended, much faster than online estimation):

```bash
python tools/gen_depth.py --config configs/bdganet_rlfdb.yaml --split train --write_pseudo_gt
python tools/gen_depth.py --config configs/bdganet_rlfdb.yaml --split test
```

`--write_pseudo_gt` also writes `depth_gt.npy` from the center-view estimation, used as pseudo GT depth when the dataset has no rendered depth. If your synthetic scenes provide rendered disparity/depth, convert it into `depth_gt.npy` instead for best fidelity.

## 3. Training

```bash
python tools/train.py --config configs/bdganet_rlfdb.yaml
```

Key training settings (as in the paper):

| Setting | Value |
| --- | --- |
| Input angular resolution | 5 × 5 |
| Patch size | 128 × 128 |
| Batch size | 1 |
| Optimizer | Adam (β1=0.9, β2=0.999) |
| Initial LR / min LR | 1e-4 → 1e-6 (cosine annealing) |
| Epochs | 150 |
| Loss balancing | λ1 = 0.1 (L_reg), λ2 = 0.5 (L_p) |

Total objective:

```
L = L_deapp + L_degeo
L_deapp = L_rec + λ2·L_p + L_cr + L_dp
L_degeo = L_dd + (L_lap + λ1·L_reg) + L_tv + L_dp + L_ap
```

- `L_rec`: L1 appearance reconstruction; `L_p`: VGG perceptual loss
- `L_cr`: contrastive reconstruction loss (pulls `I_out` toward `I_gt`, away from the rainy reference, N=5 VGG scales, weights [1/32,1/16,1/8,1/4,1])
- `L_dd`: L1 depth reconstruction; `L_lap`: Laplacian depth consistency (composite depth `D̂ = M_occ·D_de + (1−M_occ)·D_gt`)
- `L_tv`: total variation; `L_dp`: geometry-difference perception; `L_ap`: appearance-difference perception

Checkpoints, TensorBoard logs and best-model selection (by validation PSNR) are written to `experiments/BDGANet_RLFDB/`.

## 4. Evaluation

```bash
python tools/test.py --config configs/bdganet_rlfdb.yaml --ckpt experiments/BDGANet_RLFDB/best.pth
```

- Restored images are saved to `results/BDGANet_RLFDB/`.
- PSNR/SSIM are computed on the **center view** against `gt_center.png` (synthetic scenes only).

### Model complexity (Table 3)

```bash
python tools/benchmark.py --config configs/bdganet_rlfdb.yaml --size 128
```

## 5. Project structure

```
BDGANet/
├── configs/            # training configs
├── datasets/           # LF dataset loader + augmentation
├── models/             # BDGANet, DA-RERP, GAID, DGSE, AVAA, depth net, losses
├── utils/              # SAI/MacPI, metrics, Canny, Depth Anything V2 wrapper
├── tools/              # train / test / gen_depth / convert / benchmark
├── scripts/            # pretrained-weight download
└── README.md
```

## 6. Notes on reproduction

- The paper reports **34.93 dB / 0.956** on RLFDB, **34.11 dB / 0.965** on RLMB (synthetic), and **11.69 M params / 24.86 GFLOPs**. Exact reproduction depends on the dataset train/test splits, the source of GT depth, and Canny edge settings; numbers may differ slightly.
- The geometry reconstructor follows the resolution-adaptive encoder-decoder spirit of RADepth; the depth estimator (Depth Anything V2) is used **frozen** and only as an input prior.
- DA-RERP optionally consumes Canny edges (`model.use_canny: true`); the dataset loader then precomputes edges per view. Default is `false` (the mask network learns edge cues internally).
- The AVAA bottleneck is set to 16 channels with 8× spatial downsampling as a memory/quality trade-off; adjust `model.bottleneck_ch` and `downsample` if needed.

## 7. Citation

If you find this code useful, please cite:

```bibtex
@article{bdganet2026,
  title={BDGANet: Bidirectional Difference-Perception Network for Light Field Deraining with Geometry-Appearance Interaction},
  author={Zhu, Cong and Zhu, Keni and Xiao, Yao and Yan, Wenbin and Zhang, Xiaogang and Chen, Hua},
  journal={Neurocomputing},
  year={2026}
}
```

## License

MIT
