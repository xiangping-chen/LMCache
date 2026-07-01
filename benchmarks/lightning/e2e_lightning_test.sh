#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end TTFT test: vLLM + LMCache + LightningPosixBackend
#
# Launches vLLM with the Lightning Posix storage plugin, then runs the
# ttft-estimator twice (cold prefill, then warm cache-hit) and reports
# the results side by side.
#
# Usage:
#   MODEL=meta-llama/Llama-3.1-8B-Instruct ./e2e_lightning_test.sh
#   MODEL=meta-llama/Llama-3.1-70B-Instruct TP=8 ./e2e_lightning_test.sh
#
# Requires: vllm, lmcache (editable install), openai, transformers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------- tunables ----------
MODEL="${MODEL:-openai/gpt-oss-120b}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
GPU_MEM="${GPU_MEM:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131000}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-1000,2000,4000,8000,10000,20000,40000,80000,128000}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/e2e_lightning_test.yaml}"
LOG="${LOG:-/tmp/vllm_lightning_e2e.log}"
DATA_PATH="${DATA_PATH:-/mnt/lmbench/e2e_test}"

# HuggingFace cache (reuse existing model downloads)
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/data/.cache/huggingface/hub}"

# ---------- cleanup ----------
VLLM_PID=""
cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        echo "--- Shutting down vLLM (PID=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---------- clear previous slot file ----------
echo "--- Clearing Lightning data directory: $DATA_PATH"
rm -rf "$DATA_PATH"
mkdir -p "$DATA_PATH"

# ---------- start vLLM ----------
echo "--- Starting vLLM with Lightning Posix backend"
echo "    Model:       $MODEL"
echo "    TP:          $TP"
echo "    Port:        $PORT"
echo "    Config:      $CONFIG"
echo "    Data path:   $DATA_PATH"
echo "    Max len:     $MAX_MODEL_LEN"
echo "    Contexts:    $CONTEXT_LENGTHS"
echo ""

# Build CUDA_VISIBLE_DEVICES for TP
if [[ "$TP" -gt 1 ]]; then
    DEVS=$(seq -s, 0 $((TP - 1)))
    export CUDA_VISIBLE_DEVICES="$DEVS"
fi

PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE="$CONFIG" \
LMCACHE_USE_EXPERIMENTAL=True \
LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-ERROR}" \
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}" \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "$GPU_MEM" \
    -tp "$TP" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    > "$LOG" 2>&1 &
VLLM_PID=$!

# ---------- wait for server ----------
echo "--- Waiting for vLLM to be ready (polling /v1/models)..."
for i in $(seq 1 300); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "    vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died during startup. Last 50 lines of log:"
        tail -50 "$LOG"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: vLLM did not become ready in 300s. Last 50 lines of log:"
    tail -50 "$LOG"
    exit 1
fi

# ---------- run TTFT estimator ----------
TTFT_SCRIPT="${REPO_ROOT}/benchmarks/ttft-estimator/ttft-estimator.py"

echo ""
echo "========================================="
echo "  Run 1: COLD (prefill, no cache)"
echo "========================================="
python "$TTFT_SCRIPT" \
    --model "$MODEL" \
    --host localhost --port "$PORT" \
    --context-lengths "$CONTEXT_LENGTHS" \
    2>&1 | tee /tmp/ttft_cold.txt

echo ""
echo "========================================="
echo "  Run 2: WARM (cache hit from Lightning)"
echo "========================================="
python "$TTFT_SCRIPT" \
    --model "$MODEL" \
    --host localhost --port "$PORT" \
    --context-lengths "$CONTEXT_LENGTHS" \
    2>&1 | tee /tmp/ttft_warm.txt

# ---------- summary ----------
echo ""
echo "========================================="
echo "  TTFT Summary"
echo "========================================="
echo ""
printf "%-15s  %12s  %12s  %8s\n" "Context" "Cold (s)" "Warm (s)" "Speedup"
printf "%-15s  %12s  %12s  %8s\n" "-------" "--------" "--------" "-------"

paste <(grep "^Context" /tmp/ttft_cold.txt) <(grep "^Context" /tmp/ttft_warm.txt) | \
while IFS=$'\t' read -r cold_line warm_line; do
    ctx=$(echo "$cold_line" | sed 's/Context length: \([0-9]*\),.*/\1/')
    cold_ttft=$(echo "$cold_line" | sed 's/.*TTFT: //')
    warm_ttft=$(echo "$warm_line" | sed 's/.*TTFT: //')
    speedup=$(python3 -c "c=$cold_ttft; w=$warm_ttft; print(f'{c/w:.1f}x' if w > 0.001 else 'N/A')" 2>/dev/null || echo "N/A")
    printf "%-15s  %12.4f  %12.4f  %8s\n" "$ctx" "$cold_ttft" "$warm_ttft" "$speedup"
done

echo ""
echo "--- Lightning slot file stats:"
ls -lh "$DATA_PATH"/ 2>/dev/null || echo "(no files)"
echo ""
echo "--- vLLM log: $LOG"
echo "--- Done."
