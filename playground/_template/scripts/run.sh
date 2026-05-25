#!/usr/bin/env bash
# Default entry point for this experiment.
# Replace `src.<entry>` with the real module name.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$HERE/.." && pwd)"
REPO_DIR="$(cd "$EXP_DIR/../../.." && pwd)"
cd "$EXP_DIR"

echo "[run] experiment dir: $EXP_DIR"
echo "[run] TODO: replace this with 'uv run --project \"\$REPO_DIR\" python -m src.<entry>'"
