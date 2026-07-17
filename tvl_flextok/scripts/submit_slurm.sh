#!/bin/bash
# Submit TVL-FlexTok Slurm jobs with repo-local defaults.
#
# Examples:
#   tvl_flextok/scripts/submit_slurm.sh alignment
#   tvl_flextok/scripts/submit_slurm.sh alignment_4gpu DATASETS_DIR=/path/to/data
#   tvl_flextok/scripts/submit_slurm.sh reconstruction ALIGNMENT_CKPT=/path/to/checkpoint_best_joint.pth

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR_DEFAULT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB="${1:-alignment}"
shift || true

case "$JOB" in
    alignment)
        SCRIPT="$SCRIPT_DIR/train_alignment.sh"
        ;;
    alignment_4gpu|multigpu)
        SCRIPT="$SCRIPT_DIR/train_alignment_multigpu.sh"
        ;;
    reconstruction|recon)
        SCRIPT="$SCRIPT_DIR/train_reconstruction.sh"
        ;;
    prefix_vis|visualize_prefix)
        SCRIPT="$SCRIPT_DIR/visualize_prefix_recon.sh"
        ;;
    *)
        echo "Usage: $0 {alignment|alignment_4gpu|reconstruction|prefix_vis} [KEY=VALUE ...]"
        exit 2
        ;;
esac

EXPORTS="ALL,PROJECT_DIR=${PROJECT_DIR:-$PROJECT_DIR_DEFAULT}"
if [ -f "$PROJECT_DIR_DEFAULT/ckpts/tvl_enc_vittiny.pth" ]; then
    EXPORTS="$EXPORTS,STAGE1_CKPT=${STAGE1_CKPT:-$PROJECT_DIR_DEFAULT/ckpts/tvl_enc_vittiny.pth}"
fi
if [ -d "$PROJECT_DIR_DEFAULT/../datasets/tvl_dataset" ]; then
    EXPORTS="$EXPORTS,DATASETS_DIR=${DATASETS_DIR:-$PROJECT_DIR_DEFAULT/../datasets/tvl_dataset}"
fi
EXPORTS="$EXPORTS,WANDB_MODE=${WANDB_MODE:-offline}"

for kv in "$@"; do
    case "$kv" in
        *=*) EXPORTS="$EXPORTS,$kv" ;;
        *)
            echo "ERROR: extra arguments must be KEY=VALUE, got: $kv"
            exit 2
            ;;
    esac
done

echo "Submitting $JOB with:"
echo "  script: $SCRIPT"
echo "  export: $EXPORTS"
sbatch --export="$EXPORTS" "$SCRIPT"
