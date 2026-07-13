"""ARGUS-style CUPTI Activity API collection + compression on Modal.

Implements the missing third observation channel from ARGUS §4.3:
always-on GPU kernel tracing via the CUPTI Activity API (not CUDA Events,
not torch.profiler). Feeds records into the same KDE / W₁ pipeline.

Also measures CUPTI always-on step-time overhead vs baseline.

Usage::

    uv run modal run playground/argus_cupti_modal.py

Writes ``playground/argus_cupti_results.json``.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import modal
import numpy as np

GPU = os.environ.get("ARGUS_GPU", "A10G")
WARMUP = int(os.environ.get("ARGUS_CUPTI_WARMUP", "20"))
STEPS = int(os.environ.get("ARGUS_CUPTI_STEPS", "80"))

TRACER_SRC = Path(__file__).with_name("argus_cupti_tracer.cpp")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("g++", "ca-certificates")
    .pip_install("torch", "numpy", "psutil")
    .add_local_file(TRACER_SRC, remote_path="/root/argus_cupti_tracer.cpp")
    .add_local_file(
        Path(__file__).with_name("argus_demo_modal.py"),
        remote_path="/root/argus_demo_modal.py",
    )
)

app = modal.App("argus-cupti")


def build_tracer_so(out_path: Path) -> dict:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    include = f"{cuda_home}/include"
    lib = f"{cuda_home}/lib64"
    activity_h = f"{include}/cupti_activity.h"
    if not Path(activity_h).exists():
        # Some images only ship cupti.h which includes activity defs.
        activity_h = f"{include}/cupti.h"
    probe = subprocess.check_output(
        [
            "bash",
            "-lc",
            f"grep -oE 'CUpti_ActivityKernel[0-9]+' {activity_h} | sort -V | uniq | tail -1",
        ],
        text=True,
    ).strip()
    kernel_t = probe or "CUpti_ActivityKernel4"
    cmd = [
        "g++",
        "-shared",
        "-fPIC",
        "-O2",
        f"-DARGUS_KERNEL_T={kernel_t}",
        f"-I{include}",
        "/root/argus_cupti_tracer.cpp",
        f"-L{lib}",
        "-lcupti",
        "-Wl,-rpath," + lib,
        "-o",
        str(out_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"CUPTI tracer build failed ({proc.returncode}):\n"
            f"cmd={' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return {"kernel_struct": kernel_t, "so": str(out_path), "cmd": cmd}


class CuptiTracer:
    def __init__(self, so_path: Path):
        self.lib = ctypes.CDLL(str(so_path))
        self.lib.argus_cupti_start.restype = ctypes.c_int
        self.lib.argus_cupti_stop.restype = ctypes.c_int
        self.lib.argus_cupti_count.restype = ctypes.c_size_t
        self.lib.argus_cupti_clear.restype = None
        self.lib.argus_cupti_get.restype = ctypes.c_int
        self.lib.argus_cupti_kernel_struct.restype = ctypes.c_char_p

    def start(self) -> None:
        if self.lib.argus_cupti_start() != 0:
            raise RuntimeError("argus_cupti_start failed")

    def stop(self) -> None:
        if self.lib.argus_cupti_stop() != 0:
            raise RuntimeError("argus_cupti_stop failed")

    def clear(self) -> None:
        self.lib.argus_cupti_clear()

    def kernel_struct(self) -> str:
        raw = self.lib.argus_cupti_kernel_struct()
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    def records(self) -> list[tuple[str, int, float]]:
        n = int(self.lib.argus_cupti_count())
        out = []
        name_buf = ctypes.create_string_buffer(128)
        stream = ctypes.c_uint32()
        dur = ctypes.c_double()
        for i in range(n):
            rc = self.lib.argus_cupti_get(
                ctypes.c_size_t(i),
                name_buf,
                ctypes.c_size_t(128),
                ctypes.byref(stream),
                ctypes.byref(dur),
            )
            if rc != 0:
                continue
            out.append((name_buf.value.decode("utf-8", "replace"), int(stream.value), float(dur.value)))
        return out


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    return ys[len(ys) // 2]


def build_model(device, dtype):
    import torch.nn as nn

    width, depth = 512, 24
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers.extend([nn.Linear(width, width), nn.GELU()])
    layers.append(nn.LayerNorm(width))
    return nn.Sequential(*layers).to(device=device, dtype=dtype), width


def train_step(model, opt, x, *, semantics: bool) -> None:
    import torch

    if not semantics:
        out = model(x)
        loss = out.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return
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


def measure_steps(model, opt, x, *, n: int, semantics: bool = False) -> list[float]:
    import torch

    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_step(model, opt, x, semantics=semantics)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


@app.function(gpu=GPU, image=image, timeout=1800, memory=32768)
def bench() -> dict:
    import sys

    import torch

    sys.path.insert(0, "/root")
    from argus_demo_modal import compress_rank_trace, iqr_outliers, rank_deviation_scores

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    so_path = Path("/tmp/libargus_cupti_tracer.so")
    build_meta = build_tracer_so(so_path)
    tracer = CuptiTracer(so_path)

    model, width = build_model(device, dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(8, 256, width, device=device, dtype=dtype)

    # Warmup without CUPTI.
    for _ in range(WARMUP):
        train_step(model, opt, x, semantics=False)
    torch.cuda.synchronize()

    # ── Overhead: baseline vs CUPTI vs CUPTI+semantics ───────────────────
    base_ms = measure_steps(model, opt, x, n=STEPS, semantics=False)
    base_med = _median(base_ms)

    rss0 = _rss_mb()
    tracer.clear()
    tracer.start()
    cupti_ms = measure_steps(model, opt, x, n=STEPS, semantics=False)
    torch.cuda.synchronize()
    tracer.stop()
    cupti_records = tracer.records()
    rss1 = _rss_mb()
    cupti_med = _median(cupti_ms)

    tracer.clear()
    tracer.start()
    both_ms = measure_steps(model, opt, x, n=STEPS, semantics=True)
    torch.cuda.synchronize()
    tracer.stop()
    both_med = _median(both_ms)

    # ── Straggler demo on real CUPTI kernel names ────────────────────────
    # Collect a clean CUPTI window, then simulate 8 ranks with one slow rank
    # by scaling durations of the hottest GEMM-like kernels.
    tracer.clear()
    tracer.start()
    for _ in range(30):
        train_step(model, opt, x, semantics=False)
    torch.cuda.synchronize()
    tracer.stop()
    baseline_events = tracer.records()

    # Normalize kernel names a bit: strip long template noise for grouping demos.
    def short_name(name: str) -> str:
        if "gemm" in name.lower() or "cutlass" in name.lower() or "volta" in name.lower():
            return "gemm"
        if "reduce" in name.lower():
            return "reduce"
        if "elementwise" in name.lower() or "gelu" in name.lower():
            return "elementwise"
        # Keep first 40 chars for everything else.
        return name[:40]

    condensed = [(short_name(n), s, d) for n, s, d in baseline_events]
    # Pick the most frequent kernel family as the L3 target.
    counts = Counter(n for n, _, _ in condensed)
    target_kernel = counts.most_common(1)[0][0] if counts else "gemm"

    n_ranks = 8
    straggler = 5
    slow_factor = 2.8
    rng = np.random.default_rng(0)
    rank_events = []
    for r in range(n_ranks):
        ev = []
        for name, stream, dur in condensed:
            d = dur
            if r == straggler and name == target_kernel:
                d *= slow_factor * (1.0 + 0.03 * rng.standard_normal())
            else:
                d *= 1.0 + 0.02 * rng.standard_normal()
            ev.append((name, stream, max(float(d), 1e-4)))
        rank_events.append(ev)

    summaries = [compress_rank_trace(ev) for ev in rank_events]
    # Use stream from the first event of target kernel.
    target_stream = next((s for n, s, _ in condensed if n == target_kernel), 0)
    scores = rank_deviation_scores(summaries, target_kernel, target_stream)
    flagged = iqr_outliers(scores)

    top_kernels = [
        {"name": n, "count": c}
        for n, c in Counter(n for n, _, _ in baseline_events).most_common(12)
    ]

    return {
        "device": torch.cuda.get_device_name(0),
        "cupti": {
            "api": "CUPTI Activity API",
            "kind": "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
            "kernel_struct": tracer.kernel_struct(),
            "build": build_meta,
            "records_in_overhead_window": len(cupti_records),
            "records_in_straggler_window": len(baseline_events),
            "top_kernels": top_kernels,
        },
        "overhead": {
            "warmup": WARMUP,
            "steps": STEPS,
            "baseline_median_ms": base_med,
            "cupti_median_ms": cupti_med,
            "cupti_overhead_pct": 100.0 * (cupti_med / base_med - 1.0),
            "cupti_plus_semantics_median_ms": both_med,
            "cupti_plus_semantics_overhead_pct": 100.0 * (both_med / base_med - 1.0),
            "rss_delta_mb_cupti_window": rss1 - rss0,
        },
        "l3_from_cupti": {
            "target_kernel": target_kernel,
            "target_stream": target_stream,
            "n_ranks": n_ranks,
            "simulated_straggler": straggler,
            "slow_factor": slow_factor,
            "w1_deviation_scores": scores,
            "flagged_ranks": flagged,
            "compression_mean_ratio": float(
                np.mean([s["compression_ratio"] for s in summaries])
            ),
            "detected": flagged == [straggler] or straggler in flagged,
        },
        "channels": {
            "cpu_stacks": "not in this demo (py-spy; host-side)",
            "framework_semantics": "CUDA Events on fwd/bwd/opt (overhead combo)",
            "gpu_kernels": "CUPTI Activity API CONCURRENT_KERNEL (this file)",
        },
    }


@app.local_entrypoint()
def main() -> None:
    out = Path("playground/argus_cupti_results.json")
    results = bench.remote()
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")
    oh = results["overhead"]
    l3 = results["l3_from_cupti"]
    cupti = results["cupti"]
    print(f"device={results['device']} struct={cupti['kernel_struct']}")
    print(
        f"CUPTI records/window={cupti['records_in_overhead_window']} "
        f"top={cupti['top_kernels'][:3]}"
    )
    print(
        f"overhead: CUPTI {oh['cupti_overhead_pct']:+.2f}% | "
        f"CUPTI+semantics {oh['cupti_plus_semantics_overhead_pct']:+.2f}%"
    )
    print(
        f"L3 from CUPTI kernels: target={l3['target_kernel']!r} "
        f"flagged={l3['flagged_ranks']} detected={l3['detected']} "
        f"compression~{l3['compression_mean_ratio']:.0f}x"
    )
