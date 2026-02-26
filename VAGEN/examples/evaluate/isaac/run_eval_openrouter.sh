#!/bin/bash
# Run evaluation with VAGEN environments
# Usage: ./run_eval_openrouter.sh [config.yaml] [overrides...]

# TODO:
export OPENAI_API_KEY="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"
# OpenRouter (whs 'bricks')

# TODO:
export VERL_LOGGING_LEVEL=INFO
MODEL_ID="${MODEL_ID:-qwen/qwen3.5-flash-02-23}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
shift 2>/dev/null

# Compute repository root (four levels up from this examples/isaac script -> image_bricks)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
# Move to repository root so relative paths resolve correctly
cd "$REPO_ROOT"
# Device for Isaac server (can be overridden by env var DEVICE)
DEVICE="${DEVICE:-cuda:0}"
DEFAULT_OVERRIDES=(run.backend=openai backends.openai.base_url=https://openrouter.ai/api/v1 backends.openai.model=${MODEL_ID} fileroot=${REPO_ROOT}/VAGEN)

# If user provided any override that sets run.backend, respect it; otherwise prepend default.
use_default=true
for arg in "$@"; do
	if [[ "$arg" == run.backend=* ]]; then
		use_default=false
		break
	fi
done

ray stop --force
pkill -f start_isaac_server.py || true

run_cmd=(python3 -m vagen.evaluate.run_eval --config "$CONFIG")
if [ "$use_default" = true ]; then
	run_cmd+=("${DEFAULT_OVERRIDES[@]}")
fi
run_cmd+=("$@")

"${run_cmd[@]}"
