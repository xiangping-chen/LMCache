#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Profiled e2e TTFT test: captures per-retrieve breakdown from LMCache
# INFO logs (process_tokens_time vs to_gpu_time vs broadcast_time).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-openai/gpt-oss-120b}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
GPU_MEM="${GPU_MEM:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-42000}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-1000,2000,4000,8000,10000,20000,40960}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/e2e_lightning_test.yaml}"
LOG="${LOG:-/tmp/vllm_profiled_e2e.log}"
DATA_PATH="${DATA_PATH:-/mnt/lmbench/e2e_test}"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/data/.cache/huggingface/hub}"

VLLM_PID=""
cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        echo "--- Shutting down vLLM (PID=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "--- Clearing data directory: $DATA_PATH"
rm -rf "$DATA_PATH"
mkdir -p "$DATA_PATH"

echo "--- Starting vLLM (LMCACHE_LOG_LEVEL=INFO for retrieve breakdown)"
echo "    Config: $CONFIG"
echo "    Model: $MODEL, TP: $TP, Contexts: $CONTEXT_LENGTHS"

if [[ "$TP" -gt 1 ]]; then
    DEVS=$(seq -s, 0 $((TP - 1)))
    export CUDA_VISIBLE_DEVICES="$DEVS"
fi

PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE="$CONFIG" \
LMCACHE_USE_EXPERIMENTAL=True \
LMCACHE_LOG_LEVEL=INFO \
LMCACHE_INSTRUMENTATION=1 \
VLLM_LOGGING_LEVEL=ERROR \
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

echo "--- Waiting for vLLM to be ready..."
for i in $(seq 1 300); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "    vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM died. Last 30 lines:"
        tail -30 "$LOG"
        exit 1
    fi
    sleep 1
done

TTFT_SCRIPT="${REPO_ROOT}/benchmarks/ttft-estimator/ttft-estimator.py"

# ---------- COLD run ----------
echo ""
echo "========================================="
echo "  Run 1: COLD (prefill, populates cache)"
echo "========================================="
python "$TTFT_SCRIPT" \
    --model "$MODEL" \
    --host localhost --port "$PORT" \
    --context-lengths "$CONTEXT_LENGTHS" \
    2>&1 | tee /tmp/ttft_cold.txt

# Mark the log position before warm run
COLD_LOG_LINES=$(wc -l < "$LOG")
echo "--- Log has $COLD_LOG_LINES lines after cold run"

# ---------- WARM run ----------
echo ""
echo "========================================="
echo "  Run 2: WARM (cache hit)"
echo "========================================="
python "$TTFT_SCRIPT" \
    --model "$MODEL" \
    --host localhost --port "$PORT" \
    --context-lengths "$CONTEXT_LENGTHS" \
    2>&1 | tee /tmp/ttft_warm.txt

# ---------- Parse retrieve breakdown from warm-run logs ----------
echo ""
echo "========================================="
echo "  Retrieve Breakdown (from WARM run logs)"
echo "========================================="

# Extract only lines from after the cold run
tail -n +$((COLD_LOG_LINES + 1)) "$LOG" > /tmp/warm_log_only.txt

python3 - <<'PYEOF'
import re
import sys

# Parse "Retrieve request N finished: ..." lines
pattern = re.compile(
    r"Retrieve request (\d+) finished: "
    r"time_to_retrieve=([\d.]+) s, "
    r"num_tokens=(\d+), local_hit_tokens=(\d+), "
    r"process_tokens_time=([\d.]+) s, "
    r"broadcast_time=([\d.]+) s, "
    r"to_gpu_time=([\d.]+) s"
)

entries = []
with open("/tmp/warm_log_only.txt") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            entries.append({
                "req_id": int(m.group(1)),
                "total": float(m.group(2)),
                "num_tokens": int(m.group(3)),
                "hit_tokens": int(m.group(4)),
                "process_tokens": float(m.group(5)),
                "broadcast": float(m.group(6)),
                "to_gpu": float(m.group(7)),
            })

if not entries:
    print("(no retrieve breakdown found in warm logs)")
    sys.exit(0)

print(f"Found {len(entries)} retrieve events in warm run\n")

# Header
fmt = "%-8s  %10s  %10s  %12s  %10s  %10s  %10s  %6s"
print(fmt % ("ReqID", "Tokens", "HitTokens", "Total (ms)",
             "ProcTok(ms)", "ToGPU(ms)", "Bcast(ms)", "%GPU"))
print(fmt % ("-----", "------", "---------", "----------",
             "-----------", "---------", "---------", "----"))

for e in entries:
    total_ms = e["total"] * 1000
    proc_ms = e["process_tokens"] * 1000
    gpu_ms = e["to_gpu"] * 1000
    bcast_ms = e["broadcast"] * 1000
    pct_gpu = (gpu_ms / total_ms * 100) if total_ms > 0.001 else 0
    print(fmt % (
        e["req_id"], e["num_tokens"], e["hit_tokens"],
        f"{total_ms:.2f}", f"{proc_ms:.2f}", f"{gpu_ms:.2f}",
        f"{bcast_ms:.2f}", f"{pct_gpu:.0f}%"
    ))

# Summary
print()
total_proc = sum(e["process_tokens"] for e in entries)
total_gpu = sum(e["to_gpu"] for e in entries)
total_bcast = sum(e["broadcast"] for e in entries)
total_time = sum(e["total"] for e in entries)
total_tokens = sum(e["hit_tokens"] for e in entries)

print(f"Aggregate across {len(entries)} retrieves:")
print(f"  Total tokens retrieved:  {total_tokens}")
print(f"  Total retrieve time:     {total_time*1000:.1f} ms")
print(f"    process_tokens (I/O):  {total_proc*1000:.1f} ms  ({total_proc/total_time*100:.1f}%)")
print(f"    to_gpu (PCIe):         {total_gpu*1000:.1f} ms  ({total_gpu/total_time*100:.1f}%)")
print(f"    broadcast (NCCL):      {total_bcast*1000:.1f} ms  ({total_bcast/total_time*100:.1f}%)")
unaccounted = total_time - total_proc - total_gpu - total_bcast
print(f"    unaccounted:           {unaccounted*1000:.1f} ms  ({unaccounted/total_time*100:.1f}%)")

PYEOF

# ---------- Lightning I/O traces ----------
echo ""
echo "========================================="
echo "  Lightning I/O Traces (warm run)"
echo "========================================="
grep "LIGHTNING_TRACE\|LIGHTNING_PERF" /tmp/warm_log_only.txt | head -30 || echo "(no traces)"

echo ""
echo "--- Slot file stats:"
ls -lh "$DATA_PATH"/ 2>/dev/null | head -20
echo ""
echo "--- Done."
