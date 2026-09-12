#!/bin/bash
#SBATCH --job-name=t2mi-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

CKPT="${1:-results/model-final.pt}"
N_SAMPLES="${2:-1000}"

nvidia-smi
python evaluate.py \
    --checkpoint "$CKPT" \
    --n-samples "$N_SAMPLES" \
    --batch-size 16 \
    --feature 768
