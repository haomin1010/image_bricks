#!/bin/bash
# Run evaluation with VAGEN environments
# Usage: ./run_eval.sh [config.yaml] [overrides...]
export QWEN_API_KEY="sk-ba495c1895e0401ea8b3a4f3a9b5d374"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export VERL_LOGGING_LEVEL=INFO
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
shift 2>/dev/null

# Compute repository root (four levels up from this examples/isaac script -> image_bricks)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
# Move to repository root so relative paths resolve correctly
cd "$REPO_ROOT"
# Device for Isaac server (can be overridden by env var DEVICE)
DEVICE="${DEVICE:-cuda:0}"
# Default to API backend (qwen) to avoid loading local models. Users can override via extra args.
# Default overrides: use qwen backend and write outputs inside the repository
DEFAULT_OVERRIDES=(run.backend=qwen fileroot=${REPO_ROOT}/VAGEN)

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
