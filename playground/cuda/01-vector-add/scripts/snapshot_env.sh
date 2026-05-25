#!/usr/bin/env bash
# Capture a reproducibility snapshot for this experiment.
# Usage: ./scripts/snapshot_env.sh > env.txt
set -euo pipefail

echo "# Captured: $(date -Is)"
echo

echo "## OS"
uname -a || true
if [[ -r /etc/os-release ]]; then
    grep -E '^(NAME|VERSION)=' /etc/os-release || true
fi
echo

echo "## CPU"
if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E 'Model name|Socket|Core|Thread|CPU MHz' || true
fi
echo

echo "## GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
               --format=csv,noheader
else
    echo "nvidia-smi not found"
fi
echo

echo "## CUDA toolkit"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 1
else
    echo "nvcc not found"
fi
echo

echo "## Python / torch"
python - <<'PY' 2>/dev/null || echo "python/torch not importable"
import sys
print(f"python {sys.version.split()[0]}")
try:
    import torch
    print(f"torch {torch.__version__}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.version.cuda={torch.version.cuda}")
        print(f"device_count={torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {p.name} sm_{p.major}{p.minor} {p.total_memory // (1024**3)} GiB")
except Exception as e:
    print(f"torch import failed: {e}")
PY
