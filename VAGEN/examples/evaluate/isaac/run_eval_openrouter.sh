#!/bin/bash
# Run evaluation with VAGEN environments
# Usage: ./run_eval_openrouter.sh [config.yaml] [overrides...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
shift 2>/dev/null

# Compute repository root (four levels up from this examples/isaac script -> image_bricks)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Centralized config lives in repo root scripts/*.sh (per doc.md)
source "${REPO_ROOT}/scripts/template.sh"

# Back-compat: allow the old MODEL_ID env var name; template maps it to OPENROUTER_MODEL_ID.
export MODEL_ID="${MODEL_ID:-${OPENROUTER_MODEL_ID:-}}"
export VAGEN_FILEROOT="${VAGEN_FILEROOT:-${REPO_ROOT}/VAGEN}"
vagen_eval_run "$CONFIG" "openrouter" "$@"
