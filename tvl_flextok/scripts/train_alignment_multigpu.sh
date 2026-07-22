#!/bin/bash
#SBATCH --job-name=flextok-align-4gpu
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=tvl_flextok/logs/slurm/flextok_alignment_4gpu_%j.out
#SBATCH --error=tvl_flextok/logs/slurm/flextok_alignment_4gpu_%j.err

# =============================================================================
# Alignment: Multi-GPU alignment training (4x GPU with DDP).
# Uses globally gathered contrastive loss plus mandatory VAE-latent flow
# reconstruction for both modalities.
# =============================================================================

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DEFAULT_DATASETS_DIR="/viscam/u/taarush"
if [ -d "$PROJECT_DIR/../datasets/tvl_dataset" ]; then
    DEFAULT_DATASETS_DIR="$PROJECT_DIR/../datasets/tvl_dataset"
fi

STAGE1_CKPT="${STAGE1_CKPT:-/viscam/u/taarush/tvl_enc_vittiny.pth}"
DATASETS_DIR="${DATASETS_DIR:-$DEFAULT_DATASETS_DIR}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER:-peilinwu}/tvl_flextok}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_ROOT/runs/flextok_alignment_4gpu}"
LOG_NAME="${LOG_NAME:-flextok_alignment_4gpu}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"
DATASETS="${DATASETS:-ssvtp hct}"
FEATURE_MODE="${FEATURE_MODE:-sequence}"
TOKENIZER_INPUT="${TOKENIZER_INPUT:-vae_tvl}"
ENCODER_LATENT_PATCH_SIZE="${ENCODER_LATENT_PATCH_SIZE:-2}"
N_REGISTERS="${N_REGISTERS:-32}"
N_SHARED="${N_SHARED:-8}"
RECONSTRUCTION_WEIGHT="${RECONSTRUCTION_WEIGHT:-1.0}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-1.0}"
CONTINUOUS_CONTRASTIVE_WEIGHT="${CONTINUOUS_CONTRASTIVE_WEIGHT:-0.25}"
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-0.1}"
DIVERSITY_MIN_STD="${DIVERSITY_MIN_STD:-0.2}"
RECON_VIS_INTERVAL="${RECON_VIS_INTERVAL:-5}"
RECON_VIS_SAMPLES="${RECON_VIS_SAMPLES:-4}"
RECON_VIS_PREFIXES="${RECON_VIS_PREFIXES:-1 4 8 16 32 64}"
FLOW_DEPTH="${FLOW_DEPTH:-8}"
FLOW_STEPS="${FLOW_STEPS:-25}"
FLOW_CONDITION_DROPOUT="${FLOW_CONDITION_DROPOUT:-0.1}"
FLOW_GUIDANCE_SCALE="${FLOW_GUIDANCE_SCALE:-1.0}"
SAVE_LATEST="${SAVE_LATEST:-0}"
OVERFIT_ONE_SAMPLE="${OVERFIT_ONE_SAMPLE:-0}"
OVERFIT_SAMPLES="${OVERFIT_SAMPLES:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-10}"
LR="${LR:-}"
BLR="${BLR:-3e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
export WANDB_MODE="${WANDB_MODE:-offline}"
CODEC_CACHE_DIR="${CODEC_CACHE_DIR:-$PROJECT_DIR/tvl_flextok/logs/models/flextok_vae_c4}"
CODEC_LOCAL_FILES_ONLY="${CODEC_LOCAL_FILES_ONLY:-1}"
RUN_LOG_DIR="${RUN_LOG_DIR:-tvl_flextok/logs/runs/$LOG_NAME}"
ARCHIVE_ARTIFACTS="${ARCHIVE_ARTIFACTS:-1}"
WARM_START="${WARM_START:-}"
FREEZE_TOKENIZER="${FREEZE_TOKENIZER:-0}"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="${PYTHON:-python}"
fi

cd "$PROJECT_DIR" || { echo "ERROR: Could not cd to PROJECT_DIR=$PROJECT_DIR"; exit 1; }
mkdir -p "$OUTPUT_DIR" tvl_flextok/logs/slurm "$RUN_LOG_DIR" "$CODEC_CACHE_DIR"

AVAILABLE_KB=$(df -Pk "$OUTPUT_DIR" | awk 'NR==2 {print $4}')
if [ "$AVAILABLE_KB" -lt 524288 ]; then
    echo "ERROR: less than 512 MiB free on the persistent output filesystem"
    exit 1
fi

echo "========================================="
echo "TVL-FlexTok multi-GPU alignment training"
echo "Project dir: $PROJECT_DIR"
echo "Stage 1 checkpoint: $STAGE1_CKPT"
echo "Datasets dir: $DATASETS_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Feature mode: $FEATURE_MODE"
echo "Registers: $N_REGISTERS total, $N_SHARED shared"
echo "Flow reconstruction weight: $RECONSTRUCTION_WEIGHT"
echo "W&B mode: $WANDB_MODE"
echo "GPUs: $NUM_GPUS"
echo "Python: $PYTHON_BIN"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "========================================="

SAVE_LATEST_ARGS=()
if [ "$SAVE_LATEST" != "1" ]; then
    SAVE_LATEST_ARGS+=(--no_save_latest)
fi
OVERFIT_ARGS=()
if [ "$OVERFIT_ONE_SAMPLE" = "1" ]; then
    OVERFIT_ARGS+=(--overfit_one_sample)
    if [ -z "$LR" ]; then
        LR="3e-4"
        WARMUP_EPOCHS=0
    fi
fi
if [ "$OVERFIT_SAMPLES" -gt 0 ]; then
    OVERFIT_ARGS+=(--overfit_samples "$OVERFIT_SAMPLES")
    if [ -z "$LR" ]; then
        LR="3e-4"
        WARMUP_EPOCHS=0
    fi
fi
CODEC_ARGS=()
if [ "$CODEC_LOCAL_FILES_ONLY" = "1" ]; then
    CODEC_ARGS+=(--codec_local_files_only)
fi
LR_ARGS=()
if [ -n "$LR" ]; then
    LR_ARGS+=(--lr "$LR")
fi
WARM_START_ARGS=()
if [ -n "$WARM_START" ]; then
    WARM_START_ARGS+=(--warm_start "$WARM_START")
fi
if [ "$FREEZE_TOKENIZER" = "1" ]; then
    WARM_START_ARGS+=(--freeze_tokenizer)
fi

"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=$NUM_GPUS \
    tvl_flextok/train.py \
    --stage alignment \
    --feature_mode "$FEATURE_MODE" \
    --tokenizer_input "$TOKENIZER_INPUT" \
    --encoder_latent_patch_size "$ENCODER_LATENT_PATCH_SIZE" \
    --stage1_checkpoint "$STAGE1_CKPT" \
    --tactile_model vit_tiny_patch16_224 \
    --datasets_dir "$DATASETS_DIR" \
    --datasets $DATASETS \
    --output_dir "$OUTPUT_DIR" \
    --log_name "$LOG_NAME" \
    --log_dir "$RUN_LOG_DIR" \
    --hidden_dim 512 \
    --n_registers "$N_REGISTERS" \
    --n_shared "$N_SHARED" \
    --n_layers 4 \
    --n_heads 8 \
    --contrastive_weight "$CONTRASTIVE_WEIGHT" \
    --continuous_contrastive_weight "$CONTINUOUS_CONTRASTIVE_WEIGHT" \
    --diversity_weight "$DIVERSITY_WEIGHT" \
    --diversity_min_std "$DIVERSITY_MIN_STD" \
    --reconstruction_weight "$RECONSTRUCTION_WEIGHT" \
    --codec_cache_dir "$CODEC_CACHE_DIR" \
    "${CODEC_ARGS[@]}" \
    --flow_depth "$FLOW_DEPTH" \
    --flow_steps "$FLOW_STEPS" \
    --flow_condition_dropout "$FLOW_CONDITION_DROPOUT" \
    --flow_guidance_scale "$FLOW_GUIDANCE_SCALE" \
    --recon_vis_interval "$RECON_VIS_INTERVAL" \
    --recon_vis_samples "$RECON_VIS_SAMPLES" \
    --recon_vis_prefixes "$RECON_VIS_PREFIXES" \
    "${SAVE_LATEST_ARGS[@]}" \
    "${OVERFIT_ARGS[@]}" \
    "${LR_ARGS[@]}" \
    "${WARM_START_ARGS[@]}" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --blr "$BLR" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --weight_decay 0.05 \
    --num_workers "$NUM_WORKERS" \
    --seed 42

if [ "$ARCHIVE_ARTIFACTS" = "1" ]; then
    "$PROJECT_DIR/tvl_flextok/scripts/archive_training_artifacts.sh" "$OUTPUT_DIR/$LOG_NAME" "$RUN_LOG_DIR"
    echo "Alignment training complete. Joint checkpoint: $RUN_LOG_DIR/checkpoints/checkpoint_best_joint.pth"
fi
