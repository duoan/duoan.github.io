#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-sim}"
shift || true

DATA_DIR="${ARGUS_DATA:-/data}"
JOB_ID="${ARGUS_JOB_ID:-default}"
SOCK="${ARGUS_SOCKET:-${DATA_DIR}/argus.sock}"
mkdir -p "${DATA_DIR}/metrics" "${DATA_DIR}/objects"

case "${ROLE}" in
  sim)
    # End-to-end synthetic straggler job (CPU, no GPU required).
    exec python - <<'PY'
from pathlib import Path
import json, os
from argus.sim import run_synthetic_job

data = os.environ.get("ARGUS_DATA", "/data")
job = os.environ.get("ARGUS_JOB_ID", "synth")
out = run_synthetic_job(data_dir=data, job_id=job)
path = Path(data) / "results.json"
path.write_text(json.dumps(out, indent=2))
print(f"Wrote {path}")
print(
    f"detected={out['detected']} flagged={out['flagged_ranks']} "
    f"compression~{out['mean_compression_ratio']:.0f}x"
)
PY
    ;;
  processor)
    exec python - <<PY
from pathlib import Path
import os, time
from argus.processor.pipeline import Processor
from argus.storage.metrics import MetricStore
from argus.storage.objects import ObjectStore

data = Path(os.environ.get("ARGUS_DATA", "/data"))
job = os.environ.get("ARGUS_JOB_ID", "default")
sock = Path(os.environ.get("ARGUS_SOCKET", str(data / "argus.sock")))
proc = Processor(
    metrics=MetricStore(data / "metrics"),
    objects=ObjectStore(data / "objects"),
    job_id=job,
)
proc.start_unix_server(sock)
print(f"Processor listening on {sock}", flush=True)
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    proc.stop_unix_server()
PY
    ;;
  analyze|client)
    exec python -m argus.client.cli --data-dir "${DATA_DIR}/metrics" --job-id "${JOB_ID}" --out "${DATA_DIR}/diagnosis.json"
    ;;
  producer-demo)
    # Emit one window of fake kernels to the Processor Unix socket.
    exec python - <<PY
import os, time
from argus.producer.agent import TraceProducer

sock = os.environ.get("ARGUS_SOCKET", "/data/argus.sock")
rank = int(os.environ.get("ARGUS_RANK", "0"))
job = os.environ.get("ARGUS_JOB_ID", "default")
# Wait for processor socket.
for _ in range(100):
    if os.path.exists(sock):
        break
    time.sleep(0.1)
p = TraceProducer(rank=rank, job_id=job, use_cupti=False, socket_path=sock)
p.begin_window()
for i in range(60):
    p.emit_fake_kernel("gemm_fc1", 0, 0.4 + 0.02 * (i % 3))
    p.emit_fake_kernel("gelu", 0, 0.08)
    p.record_iteration(i, 12.0 + (5.0 if rank == 5 and i > 20 else 0.0))
    p.semantics.events.append(
        __import__("argus.schemas", fromlist=["PhaseEvent"]).PhaseEvent(
            "self_attention", 6.0 + (20.0 if rank == 5 and i > 20 else 0.0),
            rank=rank, step=i, group="dp",
        )
    )
resp = p.end_window()
print("sent window", resp.get("window_id"), "compression via processor socket OK")
PY
    ;;
  shell)
    exec bash
    ;;
  *)
    echo "Unknown role: ${ROLE}" >&2
    echo "Roles: sim | processor | analyze | producer-demo | shell" >&2
    exit 2
    ;;
esac
