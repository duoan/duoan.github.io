"""Overhead comparison: baseline vs ARGUS-style vs torch.profiler vs nsys.

Remeasures the ARGUS §8.2 claim on Modal A10G with a *launch-heavy* training
loop (many small eager ops), where CUPTI / profiler callback cost shows up.
Reports median step time, overhead vs baseline, and RSS growth.

Usage (from repo root)::

    uv run modal run playground/argus_overhead_modal.py

Writes ``playground/argus_overhead_results.json``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import modal

GPU = os.environ.get("ARGUS_GPU", "A10G")
WARMUP = int(os.environ.get("ARGUS_OH_WARMUP", "30"))
STEPS = int(os.environ.get("ARGUS_OH_STEPS", "200"))

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("wget", "ca-certificates", "gnupg")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb "
        "-O /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb && apt-get update "
        "&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
        "cuda-nsight-systems-12-4",
        # Ensure `nsys` is on PATH (package installs under /opt/nvidia/...).
        "ln -sf /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/bin/nsys "
        "|| ln -sf /usr/local/cuda/bin/nsys /usr/local/bin/nsys "
        "|| true",
        "nsys --version | head -5",
    )
    .pip_install("torch", "numpy", "psutil")
)

app = modal.App("argus-overhead")


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    return ys[len(ys) // 2]


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def build_launch_heavy_model(device, dtype):
    """Many small Linear layers → lots of kernel launches (profiler-sensitive)."""
    import torch.nn as nn

    width, depth = 512, 48
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers.extend([nn.Linear(width, width), nn.GELU()])
    layers.append(nn.LayerNorm(width))
    return nn.Sequential(*layers).to(device=device, dtype=dtype), width


def train_step(model, opt, x, *, semantics: bool = False) -> None:
    import torch

    if not semantics:
        out = model(x)
        loss = out.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return

    # ARGUS §4.2-style CUDA Events on semantic phase boundaries.
    e0, e1, e2, e3 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
    e0.record()
    out = model(x)
    loss = out.square().mean()
    e1.record()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    e2.record()
    opt.step()
    e3.record()
    e3.synchronize()
    _ = e0.elapsed_time(e1) + e1.elapsed_time(e2) + e2.elapsed_time(e3)


def run_config(name: str, *, warmup: int, steps: int) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    model, width = build_launch_heavy_model(device, dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch, seq = 8, 256
    x = torch.randn(batch, seq, width, device=device, dtype=dtype)

    semantics = name == "argus_semantics"
    use_profiler = name.startswith("torch_profiler")

    for _ in range(warmup):
        train_step(model, opt, x, semantics=False)
    torch.cuda.synchronize()

    rss_before = _rss_mb()
    rss_series = [rss_before]
    step_ms: list[float] = []
    wall0 = time.perf_counter()

    def timed_steps(n: int) -> None:
        for i in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            train_step(model, opt, x, semantics=semantics)
            torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - t0) * 1e3)
            if (i + 1) % 25 == 0:
                rss_series.append(_rss_mb())

    if use_profiler:
        activities = [ProfilerActivity.CUDA]
        record_shapes = False
        with_stack = False
        if name == "torch_profiler_full":
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            record_shapes = True
            with_stack = True
        # Always-on for the whole window — traces accumulate in-process (paper §8.2).
        with profile(
            activities=activities,
            record_shapes=record_shapes,
            with_stack=with_stack,
            profile_memory=False,
        ):
            timed_steps(steps)
    else:
        timed_steps(steps)

    wall_s = time.perf_counter() - wall0
    torch.cuda.synchronize()
    rss_after = _rss_mb()
    rss_series.append(rss_after)

    return {
        "name": name,
        "device": torch.cuda.get_device_name(0),
        "warmup": warmup,
        "steps": steps,
        "median_step_ms": _median(step_ms),
        "mean_step_ms": sum(step_ms) / len(step_ms),
        "p95_step_ms": sorted(step_ms)[int(0.95 * (len(step_ms) - 1))],
        "wall_s": wall_s,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
        "rss_series_mb": rss_series,
        "step_ms": step_ms,
        "ok": True,
        "error": None,
    }


NSYS_WORKER = r'''
import json, time, sys
import psutil
import torch
import torch.nn as nn

warmup, steps = int(sys.argv[1]), int(sys.argv[2])
device = torch.device("cuda")
dtype = torch.bfloat16
torch.manual_seed(0)
width, depth = 512, 48
layers = []
for _ in range(depth):
    layers.extend([nn.Linear(width, width), nn.GELU()])
layers.append(nn.LayerNorm(width))
model = nn.Sequential(*layers).to(device=device, dtype=dtype)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
x = torch.randn(8, 256, width, device=device, dtype=dtype)

def step():
    out = model(x)
    loss = out.square().mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

for _ in range(warmup):
    step()
torch.cuda.synchronize()
proc = psutil.Process()
rss_before = proc.memory_info().rss / (1024 * 1024)
step_ms, rss_series = [], [rss_before]
wall0 = time.perf_counter()
for i in range(steps):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    step_ms.append((time.perf_counter() - t0) * 1e3)
    if (i + 1) % 25 == 0:
        rss_series.append(proc.memory_info().rss / (1024 * 1024))
wall_s = time.perf_counter() - wall0
rss_after = proc.memory_info().rss / (1024 * 1024)
rss_series.append(rss_after)
ys = sorted(step_ms)
print("ARGUS_OH_JSON:" + json.dumps({
    "name": "nsys",
    "device": torch.cuda.get_device_name(0),
    "warmup": warmup,
    "steps": steps,
    "median_step_ms": ys[len(ys)//2],
    "mean_step_ms": sum(step_ms)/len(step_ms),
    "p95_step_ms": ys[int(0.95*(len(ys)-1))],
    "wall_s": wall_s,
    "rss_before_mb": rss_before,
    "rss_after_mb": rss_after,
    "rss_delta_mb": rss_after - rss_before,
    "rss_series_mb": rss_series,
    "step_ms": step_ms,
    "ok": True,
    "error": None,
}))
'''


@app.function(gpu=GPU, image=image, timeout=2400, memory=32768)
def bench() -> dict:
    import torch

    configs = [
        "baseline",
        "argus_semantics",
        "torch_profiler_cuda",
        "torch_profiler_full",
    ]
    results: list[dict] = []
    for name in configs:
        print(f"=== {name} ===", flush=True)
        try:
            results.append(run_config(name, warmup=WARMUP, steps=STEPS))
            r = results[-1]
            print(
                f"  median={r['median_step_ms']:.3f} ms  rssΔ={r['rss_delta_mb']:.1f} MB",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "error": repr(exc),
                    "median_step_ms": None,
                    "rss_delta_mb": None,
                }
            )
            print(f"  FAILED: {exc!r}", flush=True)

    nsys_path = shutil.which("nsys")
    print(f"=== nsys (path={nsys_path}) ===", flush=True)
    if not nsys_path:
        results.append(
            {
                "name": "nsys",
                "ok": False,
                "error": "nsys binary not found on image",
                "median_step_ms": None,
                "rss_delta_mb": None,
            }
        )
    else:
        with tempfile.TemporaryDirectory() as td:
            worker = Path(td) / "worker.py"
            worker.write_text(NSYS_WORKER)
            report = Path(td) / "report"
            cmd = [
                nsys_path,
                "profile",
                f"--output={report}",
                "--force-overwrite=true",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--cpuctxsw=none",
                "python",
                str(worker),
                str(WARMUP),
                str(STEPS),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
                payload = None
                for stream in (proc.stdout, proc.stderr):
                    for line in stream.splitlines():
                        if line.startswith("ARGUS_OH_JSON:"):
                            payload = json.loads(line[len("ARGUS_OH_JSON:") :])
                if payload is None:
                    results.append(
                        {
                            "name": "nsys",
                            "ok": False,
                            "error": (
                                f"exit={proc.returncode}; "
                                f"stdout_tail={proc.stdout[-1200:]!r}; "
                                f"stderr_tail={proc.stderr[-1200:]!r}"
                            ),
                            "median_step_ms": None,
                            "rss_delta_mb": None,
                            "nsys_path": nsys_path,
                        }
                    )
                else:
                    payload["nsys_path"] = nsys_path
                    payload["nsys_returncode"] = proc.returncode
                    # Report file size if present.
                    reps = list(Path(td).glob("report*"))
                    payload["nsys_report_bytes"] = sum(
                        p.stat().st_size for p in reps if p.is_file()
                    )
                    results.append(payload)
                    print(
                        f"  median={payload['median_step_ms']:.3f} ms  "
                        f"rssΔ={payload['rss_delta_mb']:.1f} MB  "
                        f"report={payload['nsys_report_bytes']} B",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "name": "nsys",
                        "ok": False,
                        "error": repr(exc),
                        "median_step_ms": None,
                        "rss_delta_mb": None,
                        "nsys_path": nsys_path,
                    }
                )

    baseline = next(r for r in results if r["name"] == "baseline")
    base_med = baseline.get("median_step_ms")
    summary = []
    for r in results:
        med = r.get("median_step_ms")
        overhead_pct = None
        if r.get("ok") and med is not None and base_med:
            overhead_pct = 100.0 * (med / base_med - 1.0)
        summary.append(
            {
                "name": r["name"],
                "ok": r.get("ok"),
                "median_step_ms": med,
                "overhead_pct": overhead_pct,
                "rss_delta_mb": r.get("rss_delta_mb"),
                "error": r.get("error"),
            }
        )

    return {
        "device": torch.cuda.get_device_name(0),
        "workload": {
            "kind": "launch_heavy_mlp",
            "width": 512,
            "depth_linear": 48,
            "batch": 8,
            "seq": 256,
            "dtype": "bfloat16",
        },
        "warmup": WARMUP,
        "steps": STEPS,
        "summary": summary,
        "runs": [
            {
                **{k: v for k, v in r.items() if k != "step_ms"},
                "n_step_samples": len(r.get("step_ms") or []),
            }
            for r in results
        ],
        "step_ms_by_config": {
            r["name"]: r.get("step_ms") for r in results if r.get("step_ms")
        },
    }


@app.local_entrypoint()
def main() -> None:
    out = Path("playground/argus_overhead_results.json")
    results = bench.remote()
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")
    print(
        f"device={results['device']} warmup={results['warmup']} steps={results['steps']} "
        f"workload={results['workload']}"
    )
    for row in results["summary"]:
        if row["ok"]:
            print(
                f"  {row['name']:22s}  median={row['median_step_ms']:.3f} ms  "
                f"overhead={row['overhead_pct']:+.2f}%  rssΔ={row['rss_delta_mb']:.1f} MB"
            )
        else:
            print(f"  {row['name']:22s}  FAILED: {row['error']}")
