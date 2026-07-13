"""Mini ARGUS-style kernel compression and straggler detection on Modal.

Collects lightweight CUDA op timings from a small training loop, clusters
durations with KDE valley detection (ARGUS §5.2), compresses to (count, p50,
p99) summaries, and flags a simulated straggler rank via Wasserstein-1
distance (ARGUS §6.2).

Usage (from repo root)::

    uv run modal run playground/argus_demo_modal.py
    ARGUS_GPU=A10G uv run modal run playground/argus_demo_modal.py

Writes ``playground/argus_demo_results.json``. Figures via ``argus_demo_figures.py``.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import modal
import numpy as np

GPU = os.environ.get("ARGUS_GPU", "A10G")

app = modal.App("argus-demo")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")


def scott_bandwidth(log_samples: np.ndarray) -> float:
    n = len(log_samples)
    sigma = float(np.std(log_samples, ddof=1)) if n > 1 else 1.0
    return 1.06 * sigma * (n ** (-0.2))


def gaussian_kde_grid(log_samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    h = scott_bandwidth(log_samples)
    if h <= 0:
        h = 1e-3
    diff = (grid[:, None] - log_samples[None, :]) / h
    return np.exp(-0.5 * diff * diff).mean(axis=1) / (h * math.sqrt(2 * math.pi))


def kde_valley_clusters(
    durations_ms: np.ndarray,
    *,
    min_samples_per_side: int = 3,
    min_log_gap: float = 0.15,
) -> list[dict]:
    """ARGUS-style KDE valley clustering on log-duration samples."""
    durations_ms = np.asarray(durations_ms, dtype=np.float64)
    durations_ms = durations_ms[durations_ms > 0]
    if len(durations_ms) == 0:
        return []

    log_d = np.log(durations_ms)
    lo, hi = float(log_d.min()), float(log_d.max())
    pad = max(0.05 * (hi - lo), 0.05)
    grid = np.linspace(lo - pad, hi + pad, 256)
    density = gaussian_kde_grid(log_d, grid)

    valleys: list[int] = []
    for i in range(1, len(grid) - 1):
        if density[i - 1] > density[i] < density[i + 1]:
            valleys.append(i)

    boundaries = [lo - pad]
    for idx in valleys:
        left = log_d[log_d < grid[idx]]
        right = log_d[log_d >= grid[idx]]
        if len(left) < min_samples_per_side or len(right) < min_samples_per_side:
            continue
        if boundaries and (grid[idx] - boundaries[-1]) < min_log_gap:
            continue
        boundaries.append(float(grid[idx]))
    boundaries.append(hi + pad)

    clusters: list[dict] = []
    for a, b in zip(boundaries, boundaries[1:], strict=False):
        mask = (log_d >= a) & (log_d < b)
        chunk = durations_ms[mask]
        if len(chunk) == 0:
            continue
        clusters.append(
            {
                "count": int(len(chunk)),
                "p50_ms": float(np.percentile(chunk, 50)),
                "p99_ms": float(np.percentile(chunk, 99)),
                "log_lo": a,
                "log_hi": b,
            }
        )
    return clusters


def lognormal_mixture_cdf(x: np.ndarray, clusters: list[dict]) -> np.ndarray:
    total = sum(c["count"] for c in clusters)
    if total == 0:
        return np.zeros_like(x)
    cdf = np.zeros_like(x, dtype=np.float64)
    z99 = 2.326
    for c in clusters:
        w = c["count"] / total
        mu = math.log(c["p50_ms"])
        sigma = max((math.log(c["p99_ms"]) - mu) / z99, 1e-6)
        z = (np.log(np.maximum(x, 1e-12)) - mu) / sigma
        cdf += w * 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2)))
    return np.clip(cdf, 0.0, 1.0)


def wasserstein1(cdf_a: np.ndarray, cdf_b: np.ndarray, xs: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(cdf_a - cdf_b), xs))


def compress_rank_trace(events: list[tuple[str, int, float]]) -> dict:
    """Group events by (kernel, stream) and cluster each group."""
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for name, stream, dur_ms in events:
        grouped[(name, stream)].append(dur_ms)

    compressed = []
    raw_bytes = 0
    for (name, stream), durs in sorted(grouped.items()):
        arr = np.asarray(durs, dtype=np.float64)
        raw_bytes += len(arr) * 24  # rough event size
        clusters = kde_valley_clusters(arr)
        if not clusters:
            continue
        compressed.append(
            {
                "kernel": name,
                "stream": stream,
                "samples": int(len(arr)),
                "clusters": clusters,
            }
        )

    summary_bytes = sum(len(c["clusters"]) * 24 for c in compressed)
    ratio = raw_bytes / max(summary_bytes, 1)
    return {
        "groups": compressed,
        "raw_event_bytes_est": raw_bytes,
        "summary_bytes_est": summary_bytes,
        "compression_ratio": ratio,
    }


def rank_deviation_scores(rank_summaries: list[dict], kernel: str, stream: int) -> list[float]:
    """Mean W1 from each rank to all others for one (kernel, stream) pair."""
    cdfs = []
    xs = np.logspace(-3, 1.5, 400)  # ms
    for rank in rank_summaries:
        clusters = []
        for g in rank["groups"]:
            if g["kernel"] == kernel and g["stream"] == stream:
                clusters = g["clusters"]
                break
        cdfs.append(lognormal_mixture_cdf(xs, clusters))

    n = len(cdfs)
    scores = []
    for i in range(n):
        dists = [wasserstein1(cdfs[i], cdfs[j], xs) for j in range(n) if j != i]
        scores.append(float(np.mean(dists)) if dists else 0.0)
    return scores


def iqr_outliers(
    scores: list[float],
    alpha: float = 1.5,
    min_ratio_vs_median: float = 2.0,
) -> list[int]:
    """IQR fence plus a relative-elevation gate.

    With only a few ranks, healthy scores sit in a tight band and ordinary
    timing jitter can trip Tukey's fence. Require both: above the upper fence
    *and* at least ``min_ratio_vs_median`` times the median score.
    """
    arr = np.asarray(scores, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    fence = q3 + alpha * (q3 - q1)
    med = float(np.median(arr))
    floor = med * min_ratio_vs_median if med > 0 else fence
    return [i for i, s in enumerate(scores) if s > fence and s >= floor]


@app.function(gpu=GPU, image=image, timeout=600)
def bench() -> dict:
    import torch
    import torch.nn as nn

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)

    torch.manual_seed(0)
    hidden, seq, batch = 2048, 512, 4
    model = nn.Sequential(
        nn.Linear(hidden, hidden * 4),
        nn.GELU(),
        nn.Linear(hidden * 4, hidden),
        nn.LayerNorm(hidden),
    ).to(device=device, dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def timed_op(name: str, stream_id: int, fn) -> float:
        stream = torch.cuda.Stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            fn()
            end.record(stream)
        end.synchronize()
        return name, stream_id, start.elapsed_time(end)

    # Warm up, then collect repeated op timings — mimics per-window kernel events.
    baseline_events: list[tuple[str, int, float]] = []
    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.bfloat16)
    for _ in range(20):
        out = model(x)
        loss = out.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    for _ in range(120):
        out = model(x)
        loss = out.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        h1 = model[0](x)
        baseline_events.append(timed_op("gemm_fc1", 0, lambda: model[0](x)))
        baseline_events.append(timed_op("gelu", 0, lambda h1=h1: model[1](h1)))
        h2 = model[1](h1)
        baseline_events.append(timed_op("gemm_fc2", 0, lambda h2=h2: model[2](h2)))
        h3 = model[2](h2)
        baseline_events.append(timed_op("layernorm", 1, lambda h3=h3: model[3](h3)))

    # Simulate 8 DP ranks: rank 5 is a compute straggler on GEMM kernels.
    n_ranks = 8
    straggler = 5
    slow_factor = 2.8
    rank_events: list[list[tuple[str, int, float]]] = []
    for r in range(n_ranks):
        events = []
        for name, stream, dur in baseline_events:
            if r == straggler and name.startswith("gemm"):
                dur *= slow_factor * (1.0 + 0.03 * np.random.randn())
            else:
                dur *= 1.0 + 0.02 * np.random.randn()
            events.append((name, stream, max(dur, 0.01)))
        rank_events.append(events)

    rank_summaries = [compress_rank_trace(ev) for ev in rank_events]
    avg_ratio = float(np.mean([rs["compression_ratio"] for rs in rank_summaries]))

    # L3-style detection on the main GEMM kernel.
    target_kernel, target_stream = "gemm_fc1", 0
    scores = rank_deviation_scores(rank_summaries, target_kernel, target_stream)
    flagged = iqr_outliers(scores)

    return {
        "device": device_name,
        "n_ranks": n_ranks,
        "simulated_straggler": straggler,
        "slow_factor": slow_factor,
        "baseline_events_per_rank": len(baseline_events),
        "compression": {
            "mean_ratio": avg_ratio,
            "per_rank_ratio": [rs["compression_ratio"] for rs in rank_summaries],
            "example_rank0": {
                "raw_bytes_est": rank_summaries[0]["raw_event_bytes_est"],
                "summary_bytes_est": rank_summaries[0]["summary_bytes_est"],
                "groups": rank_summaries[0]["groups"],
            },
        },
        "l3_detection": {
            "kernel": target_kernel,
            "stream": target_stream,
            "w1_deviation_scores": scores,
            "flagged_ranks": flagged,
        },
        "kde_demo": {
            "kernel": "gemm_fc1",
            "stream": 0,
            "durations_ms": [
                d for n, s, d in rank_events[0] if n == "gemm_fc1" and s == 0
            ],
            "clusters": next(
                g["clusters"]
                for g in rank_summaries[0]["groups"]
                if g["kernel"] == "gemm_fc1" and g["stream"] == 0
            ),
        },
    }


@app.local_entrypoint()
def main() -> None:
    out = Path("playground/argus_demo_results.json")
    try:
        results = bench.remote()
        source = "modal"
    except Exception as exc:  # noqa: BLE001 — Modal auth/GPU may be unavailable
        print(f"Modal run unavailable ({exc}); falling back to local synthetic demo.")
        results = local_bench()
        source = "local_synthetic"
    results["source"] = source
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")
    det = results["l3_detection"]
    print(
        f"compression ~{results['compression']['mean_ratio']:.0f}x | "
        f"flagged ranks {det['flagged_ranks']} (true straggler={results['simulated_straggler']})"
    )


def local_bench() -> dict:
    """CPU-only fallback: synthetic kernel timings, same ARGUS-style analysis."""
    rng = np.random.default_rng(0)
    n_steps = 120
    baseline_events: list[tuple[str, int, float]] = []

    def sample_lognorm(median_ms: float, p99_ratio: float, n: int) -> np.ndarray:
        mu = math.log(median_ms)
        sigma = (math.log(median_ms * p99_ratio) - mu) / 2.326
        return rng.lognormal(mu, sigma, n)

    for _ in range(n_steps):
        for dur in sample_lognorm(0.42, 1.35, 1):
            baseline_events.append(("gemm_fc1", 0, float(dur)))
        for dur in sample_lognorm(0.08, 1.4, 1):
            baseline_events.append(("gelu", 0, float(dur)))
        for dur in sample_lognorm(0.38, 1.32, 1):
            baseline_events.append(("gemm_fc2", 0, float(dur)))
        for dur in sample_lognorm(0.05, 1.5, 1):
            baseline_events.append(("layernorm", 1, float(dur)))

    n_ranks = 8
    straggler = 5
    slow_factor = 2.8
    rank_events: list[list[tuple[str, int, float]]] = []
    for r in range(n_ranks):
        events = []
        for name, stream, dur in baseline_events:
            if r == straggler and name.startswith("gemm"):
                dur *= slow_factor * (1.0 + 0.03 * rng.standard_normal())
            else:
                dur *= 1.0 + 0.02 * rng.standard_normal()
            events.append((name, stream, max(float(dur), 0.01)))
        rank_events.append(events)

    rank_summaries = [compress_rank_trace(ev) for ev in rank_events]
    avg_ratio = float(np.mean([rs["compression_ratio"] for rs in rank_summaries]))
    target_kernel, target_stream = "gemm_fc1", 0
    scores = rank_deviation_scores(rank_summaries, target_kernel, target_stream)
    flagged = iqr_outliers(scores)

    return {
        "device": "synthetic (local fallback)",
        "n_ranks": n_ranks,
        "simulated_straggler": straggler,
        "slow_factor": slow_factor,
        "baseline_events_per_rank": len(baseline_events),
        "compression": {
            "mean_ratio": avg_ratio,
            "per_rank_ratio": [rs["compression_ratio"] for rs in rank_summaries],
            "example_rank0": {
                "raw_bytes_est": rank_summaries[0]["raw_event_bytes_est"],
                "summary_bytes_est": rank_summaries[0]["summary_bytes_est"],
                "groups": rank_summaries[0]["groups"],
            },
        },
        "l3_detection": {
            "kernel": target_kernel,
            "stream": target_stream,
            "w1_deviation_scores": scores,
            "flagged_ranks": flagged,
        },
        "kde_demo": {
            "kernel": "gemm_fc1",
            "stream": 0,
            "durations_ms": [
                d for n, s, d in rank_events[0] if n == "gemm_fc1" and s == 0
            ],
            "clusters": next(
                g["clusters"]
                for g in rank_summaries[0]["groups"]
                if g["kernel"] == "gemm_fc1" and g["stream"] == 0
            ),
        },
    }


if __name__ == "__main__":
    out = Path("playground/argus_demo_results.json")
    results = local_bench()
    results["source"] = "local_synthetic"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")
