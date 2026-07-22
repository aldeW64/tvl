#!/bin/bash

set -e
set -o pipefail

RUN_OUTPUT_DIR="${1:?usage: archive_training_artifacts.sh RUN_OUTPUT_DIR RUN_LOG_DIR}"
RUN_LOG_DIR="${2:?usage: archive_training_artifacts.sh RUN_OUTPUT_DIR RUN_LOG_DIR}"
CHECKPOINT_DIR="$RUN_LOG_DIR/checkpoints"

mkdir -p "$CHECKPOINT_DIR" "$RUN_LOG_DIR/reconstructions" "$RUN_LOG_DIR/generations"
shopt -s nullglob
checkpoints=("$RUN_OUTPUT_DIR"/checkpoint_best_*.pth)
if [ "${#checkpoints[@]}" -eq 0 ]; then
    echo "ERROR: no best checkpoints found in $RUN_OUTPUT_DIR" >&2
    exit 1
fi

if [ -d "$RUN_OUTPUT_DIR/generations" ]; then
    generations=("$RUN_OUTPUT_DIR"/generations/*)
    if [ "${#generations[@]}" -gt 0 ]; then
        mv -v "${generations[@]}" "$RUN_LOG_DIR/generations/"
    fi
fi

for checkpoint in "${checkpoints[@]}"; do
    mv -v "$checkpoint" "$CHECKPOINT_DIR/"
done

if [ -d "$RUN_OUTPUT_DIR/reconstructions" ]; then
    reconstructions=("$RUN_OUTPUT_DIR"/reconstructions/*)
    if [ "${#reconstructions[@]}" -gt 0 ]; then
        mv -v "${reconstructions[@]}" "$RUN_LOG_DIR/reconstructions/"
    fi
fi

if [ -f "$RUN_OUTPUT_DIR/log.txt" ]; then
    mv -v "$RUN_OUTPUT_DIR/log.txt" "$RUN_LOG_DIR/training_metrics.jsonl"
fi

sync
printf 'source=%s\narchived_at=%s\n' "$RUN_OUTPUT_DIR" "$(date --iso-8601=seconds)" \
    > "$RUN_LOG_DIR/archive_manifest.txt"
echo "Archived best checkpoints, reconstructions, and generations under $RUN_LOG_DIR"
