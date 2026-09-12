#!/bin/bash
#SBATCH --job-name=t2mi-train
#SBATCH --partition=gpu      
#SBATCH --gres=gpu:h200:1 
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

nvidia-smi

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA unavailable — wrong torch version")
print(f"{torch.cuda.get_device_name(0)} | cuda {torch.version.cuda} | bf16 {torch.cuda.is_bf16_supported()}")
PY

python - <<'PY'
from config import Config
from dataset import IUXrayDataset
kw = dict(image_size=Config.image_size, split_ratio=Config.split_ratio,
          max_samples=Config.max_samples, frontal_only=Config.frontal_only)
tr = IUXrayDataset(Config.data_dir, is_train=True, **kw)
va = IUXrayDataset(Config.data_dir, is_train=False, **kw)
print(f"train {len(tr)} | val {len(va)} | val unique captions {len({s['caption'] for s in va.samples})}")
PY

python train.py
