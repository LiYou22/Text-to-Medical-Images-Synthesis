#!/bin/bash
#SBATCH --job-name=t2mi-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00           # 2000 张 x 250 步 DDIM 约 2.2h，留出余量
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

CKPT="${1:-results/checkpoints/model-final.pt}"
N_SAMPLES="${2:-2000}"   # >=2048 维特征需要足够样本估协方差，1000 会秩亏

nvidia-smi
python evaluate.py \
    --checkpoint "$CKPT" \
    --n-samples "$N_SAMPLES" \
    --batch-size 16 \
    --features 2048 768 \
    --save-samples results/eval-samples
