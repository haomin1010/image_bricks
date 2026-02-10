#!/usr/bin/env bash
# ============================================================================
# BrickIsaac – Evaluation with Qwen2.5-VL-3B via sglang
#
# Workflow:
#   1. Launch sglang inference server (background)
#   2. Wait for server to be ready
#   3. Run evaluation against BrickIsaac mock environment
#   4. Auto-cleanup server on exit
#
# Usage:
#   bash eval_qwen25_vl_3b.sh                         # default settings
#   MODEL_PATH=/path/to/checkpoint bash eval_qwen25_vl_3b.sh  # custom model
#   CUDA_VISIBLE_DEVICES=0 bash eval_qwen25_vl_3b.sh  # specify GPU
# ============================================================================
set -euo pipefail

# ---------- Paths ----------
fileroot="${fileroot:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-"$SCRIPT_DIR/config.yaml"}"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-"$SCRIPT_DIR/logs"}"
mkdir -p "$LOG_DIR"

# ---------- GPU selection ----------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---------- Model / Server Config ----------
MODEL_NAME="${MODEL_NAME:-"qwen25_vl_3b"}"
MODEL_PATH="${MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.80}"

DUMP_DIR="${DUMP_DIR:-"$fileroot/rollouts/brick_isaac_${MODEL_NAME}"}"
mkdir -p "$DUMP_DIR"

SERVER_LOG="${LOG_DIR}/${MODEL_NAME}_server.log"
EVAL_LOG="${LOG_DIR}/${MODEL_NAME}_eval.log"

echo "============================================"
echo "BrickIsaac Evaluation"
echo "  Model:    ${MODEL_PATH}"
echo "  GPU:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  Port:     ${PORT}"
echo "  Dump dir: ${DUMP_DIR}"
echo "============================================"

# ---------- Launch sglang Server ----------
python3 -m sglang.launch_server \
  --host 0.0.0.0 \
  --log-level warning \
  --port "${PORT}" \
  --model-path "${MODEL_PATH}" \
  --dp-size "${DP_SIZE}" \
  --tp "${TP_SIZE}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# ---------- Cleanup on exit ----------
cleanup() {
  echo "[INFO] Shutting down sglang server (PID=${SERVER_PID})..."
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
  wait "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ---------- Wait for server ----------
source "${SCRIPT_DIR}/wait_for_server.sh"
wait_for_server

# ---------- Run Evaluation ----------
cd "${fileroot}"
python -m vagen.evaluate.run_eval --config "${CONFIG}" \
  run.backend=sglang \
  backends.sglang.base_url="http://127.0.0.1:${PORT}/v1" \
  backends.sglang.model="${MODEL_PATH}" \
  experiment.dump_dir="${DUMP_DIR}" \
  fileroot="${fileroot}" \
  "$@" \
  2>&1 | tee "${EVAL_LOG}"

echo ""
echo "============================================"
echo "Evaluation complete!"
echo "  Results: ${DUMP_DIR}"
echo "  Log:     ${EVAL_LOG}"
echo "============================================"

