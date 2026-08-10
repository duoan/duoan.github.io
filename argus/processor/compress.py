"""Online KDE valley clustering + compression (§5.2)."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from argus.schemas import ClusterStats, KernelEvent, KernelGroupSummary, RankWindowSummary


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
) -> list[ClusterStats]:
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

    clusters: list[ClusterStats] = []
    for a, b in zip(boundaries, boundaries[1:], strict=False):
        mask = (log_d >= a) & (log_d < b)
        chunk = durations_ms[mask]
        if len(chunk) == 0:
            continue
        clusters.append(
            ClusterStats(
                count=int(len(chunk)),
                p50_ms=float(np.percentile(chunk, 50)),
                p99_ms=float(np.percentile(chunk, 99)),
                log_lo=a,
                log_hi=b,
            )
        )
    return clusters


def compress_kernel_events(
    events: list[KernelEvent],
    *,
    job_id: str,
    rank: int,
    window_id: str,
) -> RankWindowSummary:
    """Group by (kernel, stream) and emit cluster triples (count, p50, p99)."""
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for ev in events:
        grouped[(ev.name, ev.stream)].append(ev.duration_ms)

    groups: list[KernelGroupSummary] = []
    raw_bytes = 0
    for (name, stream), durs in sorted(grouped.items()):
        arr = np.asarray(durs, dtype=np.float64)
        raw_bytes += len(arr) * 24
        clusters = kde_valley_clusters(arr)
        if not clusters:
            continue
        groups.append(
            KernelGroupSummary(
                kernel=name,
                stream=stream,
                samples=int(len(arr)),
                clusters=clusters,
            )
        )

    summary_bytes = sum(len(g.clusters) * 24 for g in groups)
    return RankWindowSummary(
        job_id=job_id,
        rank=rank,
        window_id=window_id,
        groups=groups,
        raw_event_bytes_est=raw_bytes,
        summary_bytes_est=summary_bytes,
    )
