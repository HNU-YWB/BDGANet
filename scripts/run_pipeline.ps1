# Example pipeline (PowerShell):
#   1. convert your .mat datasets, 2. download depth weights,
#   3. precompute depth priors, 4. train, 5. evaluate.
$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"

Write-Host "[1/5] downloading Depth Anything V2 (vitb) ..."
python scripts/download_pretrained.py --encoder vitb

Write-Host "[2/5] precomputing depth priors (train) ..."
python tools/gen_depth.py --config configs/bdganet_rlfdb.yaml --split train --write_pseudo_gt

Write-Host "[3/5] precomputing depth priors (test) ..."
python tools/gen_depth.py --config configs/bdganet_rlfdb.yaml --split test

Write-Host "[4/5] training ..."
python tools/train.py --config configs/bdganet_rlfdb.yaml

Write-Host "[5/5] evaluating ..."
python tools/test.py --config configs/bdganet_rlfdb.yaml --ckpt experiments/BDGANet_RLFDB/best.pth
