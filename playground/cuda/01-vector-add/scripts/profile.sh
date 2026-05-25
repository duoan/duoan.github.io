#!/usr/bin/env bash
# Profile the vector-add bench:
#   - nsys for whole-process timeline -> results/nsys-report.nsys-rep
#   - ncu for kernel metrics on the largest size -> results/ncu-report.ncu-rep
# Both files are git-ignored by default; check them in only if a particular
# trace is worth preserving.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$HERE/.." && pwd)"
REPO_DIR="$(cd "$EXP_DIR/../../.." && pwd)"
cd "$EXP_DIR"

mkdir -p results

if command -v nsys >/dev/null 2>&1; then
    echo "[nsys] tracing ..."
    nsys profile \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --output=results/nsys-report \
        uv run --project "$REPO_DIR" python -m src.vector_add
else
    echo "[nsys] not found, skipping"
fi

if command -v ncu >/dev/null 2>&1; then
    echo "[ncu] capturing kernel metrics ..."
    ncu \
        --target-processes all \
        --kernel-name regex:vec_add_kernel \
        --launch-skip 5 --launch-count 5 \
        --set full \
        --export results/ncu-report \
        --force-overwrite \
        uv run --project "$REPO_DIR" python -m src.vector_add
else
    echo "[ncu] not found, skipping"
fi
