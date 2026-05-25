#!/usr/bin/env bash
# Wrap run.sh in nsys to capture an NCCL-aware system trace.
#   ./scripts/profile.sh
#   NPROC=4 ./scripts/profile.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$HERE/.." && pwd)"
REPO_DIR="$(cd "$EXP_DIR/../../.." && pwd)"
cd "$EXP_DIR"

NPROC="${NPROC:-2}"
mkdir -p results

if ! command -v nsys >/dev/null 2>&1; then
    echo "[nsys] not found, skipping"
    exit 0
fi

echo "[nsys] tracing world_size=$NPROC ..."
nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt,nccl \
    --output="results/nsys-allreduce-ws${NPROC}" \
    uv run --project "$REPO_DIR" \
        torchrun --standalone --nproc_per_node="$NPROC" -m src.allreduce_bench
