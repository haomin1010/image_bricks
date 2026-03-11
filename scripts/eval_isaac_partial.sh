#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/template.sh"

cfg="${VAGEN_EVAL_CONFIG_DEFAULT:-"$REPO_ROOT/VAGEN/examples/evaluate/isaac/config_partial.yaml"}"
if vagen__first_arg_is_config_path "${1:-}"; then
  cfg="$1"
  shift
fi

export VAGEN_EVAL_BACKEND_DEFAULT="${VAGEN_EVAL_BACKEND_DEFAULT:-qwen}"

vagen_eval_run "$cfg" "$VAGEN_EVAL_BACKEND_DEFAULT" "$@"
