"""Local end-to-end synthetic stack (no Docker / Modal / GPU)."""

from __future__ import annotations

from argus.client.cli import run_client
from argus.sim import run_synthetic_job


def test_synthetic_job_detects_straggler(tmp_path):
    out = run_synthetic_job(
        data_dir=str(tmp_path),
        job_id="e2e",
        n_ranks=8,
        n_steps=40,
        straggler=5,
        onset=15,
    )
    assert out["detected"] is True
    assert 5 in out["flagged_ranks"]
    assert out["mean_compression_ratio"] > 1.0
    assert out["n_perfetto_objects"] == 8

    # FT-Client-lite against the metric store written by the job.
    report = run_client(tmp_path / "metrics", job_id="e2e")
    assert 5 in report["flagged_ranks"]
