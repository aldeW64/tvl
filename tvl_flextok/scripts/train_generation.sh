#!/bin/bash
#SBATCH --job-name=flextok-gen
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=tvl_flextok/logs/slurm/flextok_generation_%j.out
#SBATCH --error=tvl_flextok/logs/slurm/flextok_generation_%j.err

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER:-peilinwu}/tvl_flextok}"
STAGE1_CKPT="${STAGE1_CKPT:-$PROJECT_DIR/ckpts/tvl_enc_vittiny.pth}"
ALIGNMENT_CKPT="${ALIGNMENT_CKPT:?Set ALIGNMENT_CKPT to a corrected alignment checkpoint}"
DATASETS_DIR="${DATASETS_DIR:-$PROJECT_DIR/../datasets/tvl_dataset}"
DATASETS="${DATASETS:-ssvtp hct}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/runs/flextok_generation}"
LOG_NAME="${LOG_NAME:-flextok_text_generation}"
RUN_LOG_DIR="${RUN_LOG_DIR:-tvl_flextok/logs/runs/$LOG_NAME}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM_ITER="${ACCUM_ITER:-1}"
BLR="${BLR:-3e-4}"
LR="${LR:-}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
TEXT_CONDITION_DROPOUT="${TEXT_CONDITION_DROPOUT:-0.1}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-1.0}"
GENERATION_TOP_K="${GENERATION_TOP_K:-256}"
GENERATION_GUIDANCE_SCALE="${GENERATION_GUIDANCE_SCALE:-1.0}"
RESUME_INTERVAL="${RESUME_INTERVAL:-5}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-10}}"
OVERFIT_SAMPLES="${OVERFIT_SAMPLES:-0}"
GPT_HIDDEN_DIM="${GPT_HIDDEN_DIM:-512}"
GPT_DEPTH="${GPT_DEPTH:-8}"
GPT_HEADS="${GPT_HEADS:-8}"
RECON_VIS_INTERVAL="${RECON_VIS_INTERVAL:-5}"
RECON_VIS_SAMPLES="${RECON_VIS_SAMPLES:-4}"
FLOW_STEPS="${FLOW_STEPS:-25}"
CODEC_CACHE_DIR="${CODEC_CACHE_DIR:-$PROJECT_DIR/tvl_flextok/logs/models/flextok_vae_c4}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../.venv/bin/python}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$RUN_LOG_DIR" tvl_flextok/logs/slurm "$CODEC_CACHE_DIR"
test -f "$STAGE1_CKPT" || { echo "Missing Stage-1 checkpoint: $STAGE1_CKPT"; exit 1; }
test -f "$ALIGNMENT_CKPT" || { echo "Missing alignment checkpoint: $ALIGNMENT_CKPT"; exit 1; }

OVERFIT_ARGS=()
if [ "$OVERFIT_SAMPLES" -gt 0 ]; then
    OVERFIT_ARGS+=(--overfit_samples "$OVERFIT_SAMPLES")
fi

LR_ARGS=()
if [ -n "$LR" ]; then
    LR_ARGS+=(--lr "$LR")
fi

"$PYTHON_BIN" -u tvl_flextok/train.py \
    --stage generation \
    --alignment_checkpoint "$ALIGNMENT_CKPT" \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --datasets_dir "$DATASETS_DIR" \
    --datasets $DATASETS \
    --output_dir "$OUTPUT_DIR" \
    --log_name "$LOG_NAME" \
    --log_dir "$RUN_LOG_DIR" \
    --codec_cache_dir "$CODEC_CACHE_DIR" \
    --codec_local_files_only \
    --gpt_hidden_dim "$GPT_HIDDEN_DIM" \
    --gpt_depth "$GPT_DEPTH" \
    --gpt_heads "$GPT_HEADS" \
    --text_condition_dropout "$TEXT_CONDITION_DROPOUT" \
    --generation_temperature "$GENERATION_TEMPERATURE" \
    --generation_top_k "$GENERATION_TOP_K" \
    --generation_guidance_scale "$GENERATION_GUIDANCE_SCALE" \
    --flow_steps "$FLOW_STEPS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --accum_iter "$ACCUM_ITER" \
    --blr "$BLR" \
    "${LR_ARGS[@]}" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --save_latest \
    --resume_interval "$RESUME_INTERVAL" \
    --recon_vis_interval "$RECON_VIS_INTERVAL" \
    --recon_vis_samples "$RECON_VIS_SAMPLES" \
    --recon_vis_prefixes "1 4 8 16 32 64" \
    --num_workers "$NUM_WORKERS" \
    "${OVERFIT_ARGS[@]}" \
    --seed 42

"$PROJECT_DIR/tvl_flextok/scripts/archive_training_artifacts.sh" \
    "$OUTPUT_DIR/$LOG_NAME" "$RUN_LOG_DIR"
