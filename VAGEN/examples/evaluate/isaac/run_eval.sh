#!/bin/bash
# Run evaluation with VAGEN environments
# Usage: ./run_eval.sh [config.yaml] [overrides...]
export QWEN_API_KEY="sk-ba495c1895e0401ea8b3a4f3a9b5d374"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export VERL_LOGGING_LEVEL=INFO
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
shift 2>/dev/null

cd "$SCRIPT_DIR/../.."
# Default to API backend (qwen) to avoid loading local models. Users can override via extra args.
DEFAULT_OVERRIDES=(run.backend=qwen)

# If user provided any override that sets run.backend, respect it; otherwise prepend default.
use_default=true
for arg in "$@"; do
	if [[ "$arg" == run.backend=* ]]; then
		use_default=false
		break
	fi
done

if [ "$use_default" = true ]; then
	python3 -m vagen.evaluate.run_eval --config "$CONFIG" "${DEFAULT_OVERRIDES[@]}" "$@"
else
	python3 -m vagen.evaluate.run_eval --config "$CONFIG" "$@"
fi
