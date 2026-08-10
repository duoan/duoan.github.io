"""Unit tests for L1/L2/L3 diagnosis."""

from __future__ import annotations

from argus.analysis.l1 import classify_iteration, run_l1
from argus.analysis.l2 import run_l2
from argus.analysis.l3 import detect_kernel_anomalies
from argus.processor.compress import compress_kernel_events
from argus.schemas import IterationSample, KernelEvent, PhaseEvent


def test_l1_detects_regression():
    series = [10.0] * 20 + [40.0] * 20
    cls = classify_iteration(series)
    assert cls["label"] in {"regression", "both"}
    assert cls["change_point"] is not None


def test_l2_flags_straggler():
    phases = []
    for rank in range(8):
        for step in range(20):
            dur = 6.0 if rank != 5 else 80.0
            phases.append(
                PhaseEvent("self_attention", dur, rank=rank, step=step, group="dp")
            )
    out = run_l2(phases)
    assert 5 in out["flagged_ranks"]


def test_l3_flags_slow_gemm():
    summaries = []
    for rank in range(8):
        events = []
        for _ in range(60):
            dur = 0.4 * (2.8 if rank == 5 else 1.0)
            events.append(KernelEvent("gemm_fc1", 0, dur, rank=rank))
            events.append(KernelEvent("gelu", 0, 0.08, rank=rank))
        summaries.append(
            compress_kernel_events(events, job_id="t", rank=rank, window_id="w0")
        )
    det = detect_kernel_anomalies(summaries, kernel="gemm_fc1", stream=0)
    assert 5 in det["flagged_ranks"]


def test_l1_run_aggregates_ranks():
    samples = [
        IterationSample(rank=r, step=s, duration_ms=10.0 + (30.0 if r == 1 and s > 10 else 0.0))
        for r in range(4)
        for s in range(20)
    ]
    out = run_l1(samples)
    assert out["n_steps"] == 20
    assert "global" in out
