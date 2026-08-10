"""Modal end-to-end test of the ARGUS full stack.

Runs the synthetic multi-rank straggler job inside a Modal container built from
``argus/docker/Dockerfile`` (CPU path). Optional GPU CUPTI smoke when
``ARGUS_GPU`` is set and ``--cupti`` is passed.

Usage (from repo root)::

    uv run modal run argus/modal_stack.py
    uv run modal run argus/modal_stack.py --cupti
    ARGUS_GPU=A10G uv run modal run argus/modal_stack.py --cupti

Writes ``argus/modal_results.json``. Falls back to local synthetic run if Modal
auth / network is unavailable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

GPU = os.environ.get("ARGUS_GPU", "A10G")
REPO = Path(__file__).resolve().parent.parent
# `python argus/modal_stack.py` puts argus/ on sys.path first; prefer repo root.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

app = modal.App("argus-fullstack")

# Build the same image docker-compose uses — validates Dockerfile + package layout.
cpu_image = modal.Image.from_dockerfile(
    str(REPO / "argus" / "docker" / "Dockerfile"),
    context_dir=str(REPO),
)

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("g++", "ca-certificates")
    .pip_install("torch", "numpy")
    .add_local_dir(str(REPO / "argus"), remote_path="/app/argus")
    .env({"PYTHONPATH": "/app", "ARGUS_USE_CUPTI": "1", "CUDA_HOME": "/usr/local/cuda"})
)


@app.function(image=cpu_image, timeout=600)
def run_cpu_stack() -> dict:
    import tempfile

    from argus.sim import run_synthetic_job

    with tempfile.TemporaryDirectory() as td:
        out = run_synthetic_job(data_dir=td, job_id="modal-synth")
        # Drop full nested report from the Modal return if huge — keep summary.
        report = out.pop("report")
        out["report_flagged"] = report.get("flagged_ranks", [])
        out["report_notes"] = report.get("notes", [])
        out["source"] = "modal-cpu-dockerfile"
        return out


@app.function(image=cuda_image, gpu=GPU, timeout=1200, memory=32768)
def run_cupti_smoke() -> dict:
    """Short CUPTI collect + KDE compression on a tiny training step."""
    import tempfile
    from pathlib import Path

    import torch
    import torch.nn as nn

    from argus.processor.compress import compress_kernel_events
    from argus.producer.cupti import CuptiTracer, build_tracer_so
    from argus.producer.semantics import SemanticsTracer

    assert torch.cuda.is_available()
    so = Path("/tmp/libargus_cupti_tracer.so")
    build_meta = build_tracer_so(so)
    tracer = CuptiTracer(so)
    sem = SemanticsTracer(rank=0)

    model = nn.Sequential(
        nn.Linear(512, 2048),
        nn.GELU(),
        nn.Linear(2048, 512),
    ).to(device="cuda", dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(4, 128, 512, device="cuda", dtype=torch.bfloat16)

    for _ in range(5):
        out = model(x)
        loss = out.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    tracer.clear()
    tracer.start()
    with sem.phase("forward"):
        out = model(x)
        loss = out.square().mean()
    with sem.phase("backward"):
        opt.zero_grad(set_to_none=True)
        loss.backward()
    with sem.phase("optimizer"):
        opt.step()
    torch.cuda.synchronize()
    tracer.stop()

    events = tracer.records()
    summary = compress_kernel_events(
        events, job_id="cupti-smoke", rank=0, window_id="w0"
    )
    with tempfile.TemporaryDirectory() as td:
        # Touch object/metric stores for path coverage.
        from argus.processor.pipeline import Processor
        from argus.storage.metrics import MetricStore
        from argus.storage.objects import ObjectStore

        proc = Processor(
            metrics=MetricStore(td + "/metrics"),
            objects=ObjectStore(td + "/objects"),
            job_id="cupti-smoke",
        )
        ingest = proc.ingest_window(
            rank=0,
            window_id="w0",
            kernels=events,
            phases=sem.pop_events(),
        )

    return {
        "source": "modal-cupti",
        "device": torch.cuda.get_device_name(0),
        "cupti_struct": tracer.kernel_struct(),
        "build": build_meta,
        "n_kernel_events": len(events),
        "compression_ratio": summary.compression_ratio,
        "n_groups": len(summary.groups),
        "top_kernels": sorted(
            {(e.name[:48], e.stream) for e in events},
            key=lambda x: x[0],
        )[:12],
        "ingest": {
            "raw_bytes": ingest["raw_bytes"],
            "summary_bytes": ingest["summary_bytes"],
            "perfetto_key": ingest["perfetto_key"],
        },
    }


def _local_synthetic() -> dict:
    import tempfile

    from argus.sim import run_synthetic_job

    with tempfile.TemporaryDirectory() as td:
        results = run_synthetic_job(data_dir=td, job_id="local-synth")
    results["source"] = "local_synthetic_fallback"
    results["report_flagged"] = results["report"]["flagged_ranks"]
    results["report_notes"] = results["report"]["notes"]
    del results["report"]
    return results


def _write_results(results: dict) -> None:
    out_path = Path("argus/modal_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
    if "detected" in results:
        print(
            f"detected={results['detected']} flagged={results.get('flagged_ranks')} "
            f"compression~{results.get('mean_compression_ratio', 0):.0f}x "
            f"source={results.get('source')}"
        )
    else:
        print(
            f"cupti events={results.get('n_kernel_events')} "
            f"compression~{results.get('compression_ratio', 0):.0f}x "
            f"device={results.get('device')}"
        )


@app.local_entrypoint()
def main(cupti: bool = False) -> None:
    try:
        results = run_cupti_smoke.remote() if cupti else run_cpu_stack.remote()
    except Exception as exc:  # noqa: BLE001
        print(f"Modal unavailable ({exc}); running local synthetic fallback.")
        results = _local_synthetic()
    _write_results(results)


if __name__ == "__main__":
    # Plain `uv run python argus/modal_stack.py` — no Modal token required.
    _write_results(_local_synthetic())
