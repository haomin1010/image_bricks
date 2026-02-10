#!/bin/bash
# ============================================================================
# Quick evaluation runner (no auto server launch)
#
# Use this if you already have an inference server running, or want to use
# an API backend (openai / claude / together).
#
# Usage:
#   # With a running sglang server:
#   ./run_eval.sh
#
#   # With OpenAI API:
#   OPENAI_API_KEY=sk-xxx ./run_eval.sh run.backend=openai
#
#   # With custom config overrides:
#   ./run_eval.sh envs.0.n_envs=32 run.max_concurrent_jobs=8
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
shift 2>/dev/null || true

cd "$SCRIPT_DIR/../.."
python -m vagen.evaluate.run_eval --config "$CONFIG" "$@"

