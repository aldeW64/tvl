#!/bin/bash
#SBATCH --job-name=flextok-gen-vis
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=tvl_flextok/logs/slurm/flextok_generation_vis_%j.out
#SBATCH --error=tvl_flextok/logs/slurm/flextok_generation_vis_%j.err

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CHECKPOINT="${GENERATION_CKPT:?Set GENERATION_CKPT to a generation checkpoint}"
STAGE1_CKPT="${STAGE1_CKPT:-$PROJECT_DIR/ckpts/tvl_enc_vittiny.pth}"
DATASETS_DIR="${DATASETS_DIR:-$PROJECT_DIR/../datasets/tvl_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/tvl_flextok/logs/manual_generation_visualization}"
CODEC_CACHE_DIR="${CODEC_CACHE_DIR:-$PROJECT_DIR/tvl_flextok/logs/models/flextok_vae_c4}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../.venv/bin/python}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" tvl_flextok/logs/slurm
"$PYTHON_BIN" -u tvl_flextok/visualize_generation.py \
    --checkpoint "$CHECKPOINT" \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --datasets_dir "$DATASETS_DIR" \
    --datasets ssvtp \
    --output_dir "$OUTPUT_DIR" \
    --codec_cache_dir "$CODEC_CACHE_DIR" \
    --codec_local_files_only \
    --overfit_samples 8 \
    --n_samples 4 \
    --flow_steps 25
