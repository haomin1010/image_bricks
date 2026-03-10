#!/bin/bash
# Run evaluation with VAGEN environments
# Usage: ./run_eval.sh [config.yaml] [overrides...]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"

# Compute repository root (four levels up from this examples/isaac script -> image_bricks)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Centralized config lives in repo root scripts/*.sh (per doc.md)
source "${REPO_ROOT}/scripts/template.sh"

if vagen__first_arg_is_config_path "${1:-}"; then
  CONFIG="$1"
  shift
fi

# Mirror previous default behavior: if caller doesn't provide run.backend=...,
# use qwen and set fileroot to ${REPO_ROOT}/VAGEN.
export VAGEN_FILEROOT="${VAGEN_FILEROOT:-${REPO_ROOT}/VAGEN}"
vagen_eval_run "$CONFIG" "qwen" "$@"
