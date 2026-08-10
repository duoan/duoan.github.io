"""L2: phase-level straggler attribution (§6.1)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from argus.schemas import PhaseEvent

# Parallelism-group-aware routing (paper Table 3, simplified).
DEFAULT_ROUTING: dict[str, str] = {
    "self_attention": "dp",
    "mlp": "dp",
    "moe_experts": "ep",
    "forward": "dp",
    "backward": "dp",
    "optimizer": "dp",
    "allreduce": "dp",
    "allgather": "dp",
    "reducescatter": "dp",
    "alltoall": "ep",
}


def _cv(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    mu = float(arr.mean())
    if mu <= 0:
        return 0.0
    return float(arr.std(ddof=1) / mu)


def _zscores(xs: list[float]) -> list[float]:
    arr = np.asarray(xs, dtype=np.float64)
    mu, sigma = float(arr.mean()), float(arr.std(ddof=1))
    if sigma <= 0:
        return [0.0] * len(xs)
    return [float((x - mu) / sigma) for x in xs]


def run_l2(
    phases: list[PhaseEvent],
    *,
    z_threshold: float = 2.0,
    cv_threshold: float = 0.15,
    routing: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Cross-rank CV / z-score on semantic phases within parallelism groups."""
    routing = routing or DEFAULT_ROUTING
    # Aggregate mean phase duration per (phase, group, rank).
    buckets: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for p in phases:
        group = p.group or routing.get(p.phase, "dp")
        buckets[(p.phase, group)][p.rank].append(p.duration_ms)

    phase_reports: dict[str, Any] = {}
    flagged: set[int] = set()
    for (phase, group), by_rank in sorted(buckets.items()):
        ranks = sorted(by_rank)
        means = [float(np.mean(by_rank[r])) for r in ranks]
        zs = _zscores(means)
        cv = _cv(means)
        hot = [ranks[i] for i, z in enumerate(zs) if z >= z_threshold]
        if cv >= cv_threshold:
            flagged.update(hot)
        phase_reports[f"{phase}@{group}"] = {
            "phase": phase,
            "group": group,
            "ranks": ranks,
            "means_ms": means,
            "z": zs,
            "cv": cv,
            "elevated_cv": cv >= cv_threshold,
            "flagged_ranks": hot if cv >= cv_threshold else [],
        }

    return {
        "phases": phase_reports,
        "flagged_ranks": sorted(flagged),
        "z_threshold": z_threshold,
        "cv_threshold": cv_threshold,
    }
