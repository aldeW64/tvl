#!/bin/bash
#SBATCH --job-name=flextok-recon
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=tvl_flextok/logs/slurm/flextok_recon_%j.out
#SBATCH --error=tvl_flextok/logs/slurm/flextok_recon_%j.err

# =============================================================================
# FlexTok reconstruction training
#
# Freeze the tokenizer and flow decoders, then train cross-modal next-register
# prediction in both directions.
#
# Usage:
#   sbatch tvl_flextok/scripts/train_reconstruction.sh
#   sbatch --export=ALL,PROJECT_DIR=/path/to/tvl,STAGE1_CKPT=/path/to/ckpt,ALIGNMENT_CKPT=/path/to/alignment.pth,DATASETS_DIR=/path/to/data \
#     tvl_flextok/scripts/train_reconstruction.sh
# =============================================================================

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DEFAULT_DATASETS_DIR="/viscam/u/taarush"
if [ -d "$PROJECT_DIR/../datasets/tvl_dataset" ]; then
    DEFAULT_DATASETS_DIR="$PROJECT_DIR/../datasets/tvl_dataset"
fi

# ---- Paths (edit these for your setup) ----
STAGE1_CKPT="${STAGE1_CKPT:-/viscam/u/taarush/tvl_enc_vittiny.pth}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER:-peilinwu}/tvl_flextok}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:-tvl_flextok/logs/runs/flextok_alignment/checkpoints/checkpoint_best_joint.pth}"
DATASETS_DIR="${DATASETS_DIR:-$DEFAULT_DATASETS_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/runs/flextok_reconstruction}"
LOG_NAME="${LOG_NAME:-flextok_reconstruction}"
DATASETS="${DATASETS:-ssvtp}"
FEATURE_MODE="${FEATURE_MODE:-sequence}"
TOKENIZER_INPUT="${TOKENIZER_INPUT:-vae_tvl}"
ENCODER_LATENT_PATCH_SIZE="${ENCODER_LATENT_PATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-10}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
CODEC_CACHE_DIR="${CODEC_CACHE_DIR:-$PROJECT_DIR/tvl_flextok/logs/models/flextok_vae_c4}"
CODEC_LOCAL_FILES_ONLY="${CODEC_LOCAL_FILES_ONLY:-1}"
RUN_LOG_DIR="${RUN_LOG_DIR:-tvl_flextok/logs/runs/$LOG_NAME}"
ARCHIVE_ARTIFACTS="${ARCHIVE_ARTIFACTS:-1}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="${PYTHON:-python}"
fi
CODEC_ARGS=()
if [ "$CODEC_LOCAL_FILES_ONLY" = "1" ]; then
    CODEC_ARGS+=(--codec_local_files_only)
fi

# ---- Setup ----
cd "$PROJECT_DIR" || { echo "ERROR: Could not cd to PROJECT_DIR=$PROJECT_DIR"; exit 1; }
mkdir -p "$OUTPUT_DIR" tvl_flextok/logs/slurm "$RUN_LOG_DIR" "$CODEC_CACHE_DIR"

# ---- Verify paths (after cd so relative paths resolve) ----
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CKPT"
    exit 1
fi

if [ ! -f "$ALIGNMENT_CKPT" ]; then
    echo "ERROR: Alignment checkpoint not found at $ALIGNMENT_CKPT"
    echo "Run Alignment first, then update ALIGNMENT_CKPT."
    exit 1
fi

echo "========================================="
echo "TVL-FlexTok autoregressive register training"
echo "Project dir: $PROJECT_DIR"
echo "Stage 1 checkpoint: $STAGE1_CKPT"
echo "Alignment checkpoint: $ALIGNMENT_CKPT"
echo "Datasets dir: $DATASETS_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Feature mode: $FEATURE_MODE"
echo "W&B mode: $WANDB_MODE"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "========================================="

# ---- Launch ----
"$PYTHON_BIN" -u tvl_flextok/train.py \
    --stage ar \
    --feature_mode "$FEATURE_MODE" \
    --tokenizer_input "$TOKENIZER_INPUT" \
    --encoder_latent_patch_size "$ENCODER_LATENT_PATCH_SIZE" \
    --alignment_checkpoint "$ALIGNMENT_CKPT" \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --tactile_model vit_tiny_patch16_224 \
    --datasets_dir "$DATASETS_DIR" \
    --datasets $DATASETS \
    --output_dir "$OUTPUT_DIR" \
    --log_name "$LOG_NAME" \
    --log_dir "$RUN_LOG_DIR" \
    --codec_cache_dir "$CODEC_CACHE_DIR" \
    "${CODEC_ARGS[@]}" \
    --hidden_dim 512 \
    --n_registers 32 \
    --n_shared 8 \
    --n_layers 4 \
    --n_heads 8 \
    --epochs 100 \
    --batch_size 32 \
    --accum_iter 8 \
    --blr 3e-4 \
    --warmup_epochs 10 \
    --num_workers "$NUM_WORKERS" \
    --seed 42

if [ "$ARCHIVE_ARTIFACTS" = "1" ]; then
    "$PROJECT_DIR/tvl_flextok/scripts/archive_training_artifacts.sh" "$OUTPUT_DIR/$LOG_NAME" "$RUN_LOG_DIR"
    echo "AR training complete. Best checkpoint: $RUN_LOG_DIR/checkpoints/checkpoint_best_ar.pth"
else
    echo "AR training complete. Best checkpoint: $OUTPUT_DIR/$LOG_NAME/checkpoint_best_ar.pth"
fi
