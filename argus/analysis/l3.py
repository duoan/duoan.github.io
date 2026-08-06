"""L3: kernel-statistics anomaly detection via W₁ (§6.2)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from argus.schemas import ClusterStats, RankWindowSummary


def lognormal_mixture_cdf(x: np.ndarray, clusters: list[ClusterStats]) -> np.ndarray:
    total = sum(c.count for c in clusters)
    if total == 0:
        return np.zeros_like(x)
    cdf = np.zeros_like(x, dtype=np.float64)
    z99 = 2.326
    for c in clusters:
        w = c.count / total
        mu = math.log(max(c.p50_ms, 1e-12))
        sigma = max((math.log(max(c.p99_ms, 1e-12)) - mu) / z99, 1e-6)
        z = (np.log(np.maximum(x, 1e-12)) - mu) / sigma
        cdf += w * 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2)))
    return np.clip(cdf, 0.0, 1.0)


def wasserstein1(cdf_a: np.ndarray, cdf_b: np.ndarray, xs: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(cdf_a - cdf_b), xs))


def iqr_outliers(
    scores: list[float],
    alpha: float = 1.5,
    min_ratio_vs_median: float = 2.0,
) -> list[int]:
    arr = np.asarray(scores, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    fence = q3 + alpha * (q3 - q1)
    med = float(np.median(arr))
    floor = med * min_ratio_vs_median if med > 0 else fence
    return [i for i, s in enumerate(scores) if s > fence and s >= floor]


def _clusters_for(
    summary: RankWindowSummary, kernel: str, stream: int
) -> list[ClusterStats]:
    for g in summary.groups:
        if g.kernel == kernel and g.stream == stream:
            return g.clusters
    return []


def detect_kernel_anomalies(
    summaries: list[RankWindowSummary],
    *,
    kernel: str,
    stream: int = 0,
    alpha: float = 1.5,
) -> dict[str, Any]:
    xs = np.logspace(-3, 1.5, 400)
    cdfs = [
        lognormal_mixture_cdf(xs, _clusters_for(s, kernel, stream)) for s in summaries
    ]
    n = len(cdfs)
    scores: list[float] = []
    matrix = np.zeros((n, n))
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            d = wasserstein1(cdfs[i], cdfs[j], xs)
            matrix[i, j] = d
            dists.append(d)
        scores.append(float(np.mean(dists)) if dists else 0.0)

    flagged_idx = iqr_outliers(scores, alpha=alpha)
    ranks = [s.rank for s in summaries]
    return {
        "kernel": kernel,
        "stream": stream,
        "ranks": ranks,
        "w1_deviation_scores": scores,
        "w1_matrix": matrix.tolist(),
        "flagged_indices": flagged_idx,
        "flagged_ranks": [ranks[i] for i in flagged_idx],
        "compression_ratios": [s.compression_ratio for s in summaries],
    }


def run_l3(
    summaries: list[RankWindowSummary],
    *,
    target_kernels: list[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Run L3 across kernels present in summaries (or an explicit target list)."""
    if not summaries:
        return {"kernels": {}, "flagged_ranks": []}

    if target_kernels is None:
        keys: set[tuple[str, int]] = set()
        for s in summaries:
            for g in s.groups:
                keys.add((g.kernel, g.stream))
        # Prefer hottest groups by sample count on rank 0-like summary.
        scored = []
        ref = summaries[0]
        for k, st in keys:
            samples = next(
                (g.samples for g in ref.groups if g.kernel == k and g.stream == st), 0
            )
            scored.append((samples, k, st))
        scored.sort(reverse=True)
        target_kernels = [(k, st) for _, k, st in scored[:8]]

    kernel_reports = {}
    flagged: set[int] = set()
    for kernel, stream in target_kernels:
        rep = detect_kernel_anomalies(summaries, kernel=kernel, stream=stream)
        key = f"{kernel}@{stream}"
        kernel_reports[key] = rep
        flagged.update(rep["flagged_ranks"])

    return {
        "kernels": kernel_reports,
        "flagged_ranks": sorted(flagged),
        "mean_compression_ratio": float(
            np.mean([s.compression_ratio for s in summaries])
        ),
    }
