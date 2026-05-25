#!/usr/bin/env bash
# Run the vector-add bandwidth bench. Outputs land in ./results/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$HERE/.." && pwd)"
REPO_DIR="$(cd "$EXP_DIR/../../.." && pwd)"
cd "$EXP_DIR"

uv run --project "$REPO_DIR" python -m src.vector_add
