"""Unit tests for KDE compression."""

from __future__ import annotations

import numpy as np
from argus.processor.compress import compress_kernel_events, kde_valley_clusters
from argus.schemas import KernelEvent


def test_kde_finds_two_modes():
    rng = np.random.default_rng(0)
    fast = rng.lognormal(np.log(0.1), 0.05, 40)
    slow = rng.lognormal(np.log(2.0), 0.05, 40)
    clusters = kde_valley_clusters(np.concatenate([fast, slow]))
    assert len(clusters) >= 2
    p50s = sorted(c.p50_ms for c in clusters)
    assert p50s[0] < 0.5
    assert p50s[-1] > 1.0


def test_compress_ratio_positive():
    events = [
        KernelEvent("gemm", 0, 0.4 + 0.01 * (i % 5), rank=0) for i in range(80)
    ] + [KernelEvent("gelu", 0, 0.08, rank=0) for _ in range(80)]
    summary = compress_kernel_events(events, job_id="t", rank=0, window_id="w0")
    assert summary.raw_event_bytes_est > summary.summary_bytes_est
    assert summary.compression_ratio > 1.0
    assert any(g.kernel == "gemm" for g in summary.groups)
