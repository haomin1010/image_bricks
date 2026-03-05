#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/template.sh"

cfg="${1:-${VAGEN_EVAL_CONFIG_DEFAULT:-"$REPO_ROOT/VAGEN/examples/evaluate/isaac/config.yaml"}}"
if [[ "${1:-}" != "" ]]; then
  shift 2>/dev/null || true
fi

export VAGEN_EVAL_BACKEND_DEFAULT="${VAGEN_EVAL_BACKEND_DEFAULT:-qwen}"

vagen_eval_run "$cfg" "$VAGEN_EVAL_BACKEND_DEFAULT" "$@"

