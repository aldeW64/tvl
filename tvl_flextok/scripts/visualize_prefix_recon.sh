#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=prefix_recon_vis
#SBATCH --output=tvl_flextok/logs/slurm/visualize_prefix_recon_%j.log

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DEFAULT_DATASETS_DIR="/viscam/u/taarush"
if [ -d "$PROJECT_DIR/../datasets/tvl_dataset" ]; then
    DEFAULT_DATASETS_DIR="$PROJECT_DIR/../datasets/tvl_dataset"
fi
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER:-peilinwu}/tvl_flextok}"
CHECKPOINT="${CHECKPOINT:-tvl_flextok/logs/runs/flextok_alignment/checkpoints/checkpoint_best_joint.pth}"
STAGE1_CKPT="${STAGE1_CKPT:-ckpts/tvl_enc_vittiny.pth}"
DATASETS_DIR="${DATASETS_DIR:-$DEFAULT_DATASETS_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-visualizations/flextok_prefix_with_s1}"
CODEC_CACHE_DIR="${CODEC_CACHE_DIR:-$PROJECT_DIR/tvl_flextok/logs/models/flextok_vae_c4}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "$PROJECT_DIR" || { echo "ERROR: Could not cd to PROJECT_DIR=$PROJECT_DIR"; exit 1; }
mkdir -p tvl_flextok/logs/slurm

"$PROJECT_DIR/../.venv/bin/python" tvl_flextok/visualize_prefix_recon.py \
    --checkpoint "$CHECKPOINT" \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --datasets_dir "$DATASETS_DIR" \
    --codec_cache_dir "$CODEC_CACHE_DIR" \
    --codec_local_files_only \
    --n_samples 4 \
    --output_dir "$OUTPUT_DIR"
