#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Drive the LocalDiskBackend vs LightningPosixBackend vs GdsBackend benchmark
# matrix on a local NVMe filesystem and collect results into a single CSV.
#
# Usage:
#   PYTHON=/path/to/venv/bin/python ./run.sh [PATH_ROOT] [CSV] [BACKEND]
#
# BACKEND: "both" (disk+lightning, default), "all" (disk+lightning+gds),
#          or a single backend name ("disk", "lightning", "gds").
#
# Requires root for --drop-caches (page-cache control on read scenarios).

set -euo pipefail

PYTHON="${PYTHON:-python}"
PATH_ROOT="${1:-/mnt/lmbench}"
CSV="${2:-results.csv}"
BACKEND="${3:-both}"
SCRIPT="$(dirname "$0")/disk_vs_posix_local.py"

# Sweep dimensions
CHUNK_MBS=(0.25 1 4 8 16 32)
NUM_KEYS=512
BATCH=32
IO_THREADS=4
ODIRECT=on

echo "Python:    $PYTHON"
echo "Path root: $PATH_ROOT"
echo "CSV:       $CSV"
echo "Backend:   $BACKEND"
echo "Matrix:    chunk_mb={${CHUNK_MBS[*]}} num_keys=$NUM_KEYS batch=$BATCH odirect=$ODIRECT"
echo

rm -f "$CSV"

for mb in "${CHUNK_MBS[@]}"; do
  # Keep per-chunk * num_keys bounded for large chunks.
  keys=$NUM_KEYS
  if (( $(echo "$mb >= 16" | bc -l) )); then keys=128; fi
  if (( $(echo "$mb >= 64" | bc -l) )); then keys=64; fi

  echo "########## chunk_mb=$mb num_keys=$keys ##########"
  "$PYTHON" "$SCRIPT" \
    --backend "$BACKEND" --scenario all \
    --path-root "$PATH_ROOT" \
    --chunk-mb "$mb" --num-keys "$keys" --batch "$BATCH" \
    --odirect "$ODIRECT" --io-threads "$IO_THREADS" \
    --drop-caches --csv "$CSV"
  echo
done

echo "Done. Results in $CSV"
