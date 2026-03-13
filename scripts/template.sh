#!/usr/bin/env bash
# Centralized config + helpers for VAGEN evaluation entrypoints.
#
# This file is meant to be sourced by other .sh scripts:
#   source "$REPO_ROOT/scripts/template.sh"
#
# Goal (per doc.md): keep *all* user-tunable parameters in scripts/*.sh,
# and let other scripts/code consume them via environment variables and
# generated OmegaConf overrides.

if [[ -n "${__IMAGE_BRICKS_TEMPLATE_SH_LOADED:-}" ]]; then
  return 0
fi
__IMAGE_BRICKS_TEMPLATE_SH_LOADED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/isaac_reward_termination.sh"

vagen__abspath() {
  local p="$1"
  (cd "$p" 2>/dev/null && pwd)
}

vagen__repo_root_from_this_file() {
  # scripts/template.sh -> repo root is one level up.
  local this_dir
  this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  (cd "$this_dir/.." && pwd)
}

vagen__export_default() {
  # Usage: vagen__export_default VAR_NAME default_value
  local var_name="$1"
  local default_value="$2"
  local current="${!var_name:-}"
  if [[ -z "$current" ]]; then
    export "${var_name}=${default_value}"
  fi
}

vagen__has_override() {
  # Usage: vagen__has_override "prefix=" "${overrides[@]}"
  local prefix="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "${prefix}"* ]]; then
      return 0
    fi
  done
  return 1
}

vagen__first_arg_is_config_path() {
  # Treat the first CLI arg as a config path only when it clearly looks like one.
  # This preserves the historical "bash ...run_eval.sh foo=bar" override style.
  local arg="${1:-}"
  if [[ -z "$arg" || "$arg" == *=* ]]; then
    return 1
  fi
  if [[ -f "$arg" ]]; then
    return 0
  fi
  case "$arg" in
    *.yaml|*.yml|*.json)
      return 0
      ;;
  esac
  return 1
}

vagen_eval_init_defaults() {
  export IMAGE_BRICKS_ROOT="${IMAGE_BRICKS_ROOT:-"$(vagen__repo_root_from_this_file)"}"
  local user_set_isaac_server_num_envs="${ISAAC_SERVER_NUM_ENVS:-}"

  # ---- Common logging ----
  vagen__export_default VERL_LOGGING_LEVEL "INFO"

  # ---- Isaac server selection ----
  # Mirrors the previous one-liner:
  #   ISAAC_CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 ISAAC_HEADLESS=1 bash ...
  vagen__export_default ISAAC_CUDA_VISIBLE_DEVICES "7"
  vagen__export_default DEVICE "cuda:0" # TODO:
  vagen__export_default ISAAC_HEADLESS "1" # single entry for Isaac headless mode: 1=headless, 0=GUI

  # ---- Isaac server argv (consumed by Python: vagen.evaluate.run_eval) ----
  vagen__export_default ISAAC_SERVER_NUM_ENVS "2" # keep aligned with eval concurrency by default
  vagen__export_default ISAAC_SERVER_TASK "multipicture_teleport_stack_from_begin"
  vagen__export_default VAGEN_MAX_CUBES "16"
  vagen__export_default ISAAC_SERVER_RECORD "0"
  vagen__export_default ISAAC_SERVER_VIDEO_LENGTH "0"
  vagen__export_default ISAAC_SERVER_VIDEO_INTERVAL "0"
  # Optional float; if empty it is omitted.
  : "${ISAAC_SERVER_IK_LAMBDA_VAL:=""}"
  # Extra raw args, e.g. '--task foo'. Headless mode is controlled only by ISAAC_HEADLESS.
  : "${ISAAC_SERVER_EXTRA_ARGS:=""}"

  # ---- Eval config + fileroot ----
  vagen__export_default VAGEN_OUTPUT_ROOT "${IMAGE_BRICKS_ROOT}/outputs"
  mkdir -p "${VAGEN_OUTPUT_ROOT}"
  vagen__export_default VAGEN_SGLANG_LOG "${VAGEN_OUTPUT_ROOT}/sglang_server.log"
  vagen__export_default VAGEN_FILEROOT "${IMAGE_BRICKS_ROOT}"
  vagen__export_default VAGEN_EVAL_CONFIG_DEFAULT "${IMAGE_BRICKS_ROOT}/VAGEN/examples/evaluate/isaac/config.yaml"

  # ---- Backend defaults ----
  # qwen: base_url can be set via env; api_key must be provided externally.
  vagen__export_default QWEN_BASE_URL "https://dashscope.aliyuncs.com/compatible-mode/v1"
  # openrouter uses the OpenAI-compatible client but needs base_url override in config.
  vagen__export_default OPENROUTER_BASE_URL "https://openrouter.ai/api/v1"
  vagen__export_default OPENROUTER_MODEL_ID "qwen/qwen2.5-pro-16k"
  # One-click reasoning toggle for backends that support extra_body thinking controls.
  vagen__export_default VAGEN_ENABLE_THINKING "1"
  vagen__export_default VAGEN_THINKING_BUDGET "2048"
  # Eval/API concurrency controls (consumed as OmegaConf overrides in template helpers).
  vagen__export_default VAGEN_EVAL_MAX_CONCURRENT_JOBS "2"
  vagen__export_default VAGEN_OPENAI_MAX_CONCURRENCY "8"
  # Keep Isaac server env slots aligned with eval concurrency unless explicitly set.
  if [[ -z "${user_set_isaac_server_num_envs}" ]]; then
    export ISAAC_SERVER_NUM_ENVS="${VAGEN_EVAL_MAX_CONCURRENT_JOBS}"
  fi

  # Back-compat for older script var names.
  if [[ -z "${OPENROUTER_MODEL_ID:-}" && -n "${MODEL_ID:-}" ]]; then
    export OPENROUTER_MODEL_ID="${MODEL_ID}"
  fi
}

vagen_eval_cleanup_runtime() {
  # Keep previous behavior by default: stop ray + kill old isaac server.
  local do_ray_stop="${VAGEN_EVAL_RAY_STOP_FORCE:-1}"
  local do_kill_isaac="${VAGEN_EVAL_KILL_OLD_ISAAC_SERVER:-1}"

  if [[ "$do_ray_stop" == "1" ]]; then
    if command -v ray >/dev/null 2>&1; then
      ray stop --force >/dev/null 2>&1 || true
    fi
  fi
  if [[ "$do_kill_isaac" == "1" ]]; then
    pkill -f start_isaac_server.py >/dev/null 2>&1 || true
  fi
}

vagen_eval_build_default_overrides() {
  # Usage: vagen_eval_build_default_overrides backend_key
  # backend_key: qwen | openrouter | openai | sglang | ...
  local backend_key="$1"

  case "$backend_key" in
    qwen)
      printf '%s\n' \
        "run.backend=qwen" \
        "fileroot=${VAGEN_FILEROOT}"
      ;;
    openrouter)
      printf '%s\n' \
        "run.backend=openai" \
        "backends.openai.base_url=${OPENROUTER_BASE_URL}" \
        "backends.openai.model=${OPENROUTER_MODEL_ID}" \
        "run.max_concurrent_jobs=${VAGEN_EVAL_MAX_CONCURRENT_JOBS}" \
        "backends.openai.max_concurrency=${VAGEN_OPENAI_MAX_CONCURRENCY}" \
        "fileroot=${VAGEN_FILEROOT}"
      ;;
    *)
      # Default: just set fileroot (matches historical usage patterns)
      printf '%s\n' "fileroot=${VAGEN_FILEROOT}"
      ;;
  esac
}

vagen_eval_run() {
  # Usage:
  #   vagen_eval_run <config_path> <backend_key> [overrides...]
  #
  # Notes:
  # - If the user provides a 'run.backend=...' override, we do NOT inject
  #   any backend-specific defaults (mirrors old run_eval.sh behavior).
  # - If the user provides a 'fileroot=...' override, we do NOT inject one.
  local cfg_path="$1"
  local backend_key="$2"
  shift 2

  vagen_eval_init_defaults

  local repo_root="${IMAGE_BRICKS_ROOT}"
  cd "$repo_root" || return 1

  local user_overrides=("$@")
  local final_overrides=()

  # Always respect user-provided fileroot; otherwise keep historical behavior:
  # fileroot is injected only when we also inject run.backend defaults.
  local user_sets_backend="0"
  if vagen__has_override "run.backend=" "${user_overrides[@]}"; then
    user_sets_backend="1"
  fi

  if [[ "$user_sets_backend" == "1" ]]; then
    final_overrides+=("${user_overrides[@]}")
  else
    # Inject defaults for this backend.
    while IFS= read -r line; do
      [[ -n "$line" ]] && final_overrides+=("$line")
    done < <(vagen_eval_build_default_overrides "$backend_key")
    final_overrides+=("${user_overrides[@]}")
  fi

  vagen_eval_cleanup_runtime

  python3 -m vagen.evaluate.run_eval --config "$cfg_path" "${final_overrides[@]}"
}
