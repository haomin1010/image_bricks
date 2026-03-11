#!/bin/bash
# Partial-view training wrapper. Reuses the main GRPO script with dataset overrides.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-brick_isaac_partial_view_grpo_qwen25vl3b_1gpu}"
export DATASET_TRAIN="${DATASET_TRAIN:-${SCRIPT_DIR}/train_isaac_partial_view_vision.yaml}"
export DATASET_VAL="${DATASET_VAL:-${SCRIPT_DIR}/val_isaac_partial_view_vision.yaml}"

bash "${SCRIPT_DIR}/train_grpo_qwen25vl3b.sh" "$@"
