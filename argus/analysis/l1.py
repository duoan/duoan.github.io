"""L1: iteration-time anomaly detection (§6.1)."""

from __future__ import annotations

from typing import Any

import numpy as np

from argus.schemas import IterationSample


def jitter_windows(
    step_ms: list[float],
    window: int = 8,
    ratio_theta: float = 2.0,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for i in range(len(step_ms) - window + 1):
        chunk = step_ms[i : i + window]
        r = max(chunk) / max(min(chunk), 1e-9)
        if r >= ratio_theta:
            hits.append({"start": i, "end": i + window - 1, "ratio": float(r)})
    return hits


def change_point(
    step_ms: list[float],
    min_ratio: float = 1.5,
) -> dict[str, Any] | None:
    n = len(step_ms)
    best: dict[str, Any] | None = None
    for t in range(5, n - 5):
        left, right = step_ms[:t], step_ms[t:]
        mu_l, mu_r = float(np.mean(left)), float(np.mean(right))
        if mu_l <= 0:
            continue
        ratio = mu_r / mu_l
        if ratio < min_ratio:
            continue
        rsd_l = float(np.std(left, ddof=1) / mu_l)
        rsd_r = float(np.std(right, ddof=1) / max(mu_r, 1e-9))
        if rsd_l > 0.35 or rsd_r > 0.35:
            continue
        cand = {"t": t, "ratio": ratio, "mu_left": mu_l, "mu_right": mu_r}
        if best is None or ratio > best["ratio"]:
            best = cand
    return best


def classify_iteration(step_ms: list[float]) -> dict[str, Any]:
    """Classify as stable / jitter / regression / both."""
    jit = jitter_windows(step_ms)
    cp = change_point(step_ms)
    has_jitter = len(jit) > 0
    has_regression = cp is not None
    if has_jitter and has_regression:
        label = "both"
    elif has_jitter:
        label = "jitter"
    elif has_regression:
        label = "regression"
    else:
        label = "stable"
    return {
        "label": label,
        "jitter_hits": jit,
        "change_point": cp,
        "median_ms": float(np.median(step_ms)) if step_ms else 0.0,
        "p95_ms": float(np.percentile(step_ms, 95)) if step_ms else 0.0,
    }


def run_l1(samples: list[IterationSample]) -> dict[str, Any]:
    """Per-rank L1 plus global (max-across-ranks) iteration series."""
    by_rank: dict[int, list[tuple[int, float]]] = {}
    for s in samples:
        by_rank.setdefault(s.rank, []).append((s.step, s.duration_ms))

    per_rank = {}
    for rank, pairs in sorted(by_rank.items()):
        pairs = sorted(pairs)
        series = [d for _, d in pairs]
        per_rank[rank] = classify_iteration(series)

    # Global iteration time ≈ max across ranks per step (synchronous barrier).
    steps = sorted({s.step for s in samples})
    global_series = []
    for step in steps:
        vals = [s.duration_ms for s in samples if s.step == step]
        global_series.append(max(vals) if vals else 0.0)
    global_cls = classify_iteration(global_series)

    anomalous_ranks = [
        r for r, c in per_rank.items() if c["label"] != "stable"
    ]
    return {
        "global": global_cls,
        "per_rank": {str(k): v for k, v in per_rank.items()},
        "anomalous_ranks": anomalous_ranks,
        "n_steps": len(steps),
    }
