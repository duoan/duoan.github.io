#!/usr/bin/env bash
# Run the all_reduce bandwidth bench. Override world size via NPROC.
#   ./scripts/run.sh
#   NPROC=4 ./scripts/run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$HERE/.." && pwd)"
REPO_DIR="$(cd "$EXP_DIR/../../.." && pwd)"
cd "$EXP_DIR"

NPROC="${NPROC:-2}"
echo "[run] world_size=$NPROC"

uv run --project "$REPO_DIR" \
    torchrun --standalone --nproc_per_node="$NPROC" -m src.allreduce_bench
