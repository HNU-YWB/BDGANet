# BDGANet: Bidirectional Depth-Guided Appearance Restoration Network for Light Field Deraining
<img width="1982" height="1117" alt="fig 2" src="https://github.com/user-attachments/assets/5ff764fb-a4e5-475d-ab50-cbeb870d8859" />


Official implementation of the paper **"BDGANet: Bidirectional Depth-Guided Appearance Restoration Network for Light Field Deraining"** 
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

The training code will be released upon paper acceptance.

## 4. Evaluation

```bash
python tools/test.py --config configs/bdganet_rlfdb.yaml --ckpt experiments/BDGANet_RLFDB/best.pth
```

- Restored images are saved to `results/BDGANet_RLFDB/`.
- PSNR/SSIM are computed on the **center view** against `gt_center.png` (synthetic scenes only).

## 4. Project structure

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

## License

MIT
