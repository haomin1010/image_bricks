#!/bin/bash
# Run evaluation with VAGEN environments
# Usage: ./run_eval_openrouter.sh [config.yaml] [overrides...]

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

# Back-compat: allow the old MODEL_ID env var name; template maps it to OPENROUTER_MODEL_ID.
export MODEL_ID="${MODEL_ID:-${OPENROUTER_MODEL_ID:-}}"
export VAGEN_FILEROOT="${VAGEN_FILEROOT:-${REPO_ROOT}/VAGEN}"
vagen_eval_run "$CONFIG" "openrouter" "$@"
