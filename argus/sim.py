"""Multi-rank synthetic training workload for end-to-end ARGUS demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from argus.analysis.diagnose import diagnose
from argus.processor.pipeline import Processor
from argus.producer.agent import TraceProducer
from argus.schemas import (
    ClusterStats,
    IterationSample,
    KernelGroupSummary,
    PhaseEvent,
    RankWindowSummary,
)
from argus.storage.metrics import MetricStore
from argus.storage.objects import ObjectStore


def _summary_from_dict(s: dict[str, Any]) -> RankWindowSummary:
    groups = [
        KernelGroupSummary(
            kernel=g["kernel"],
            stream=int(g["stream"]),
            samples=int(g["samples"]),
            clusters=[
                ClusterStats(
                    count=int(c["count"]),
                    p50_ms=float(c["p50_ms"]),
                    p99_ms=float(c["p99_ms"]),
                    log_lo=float(c.get("log_lo", 0.0)),
                    log_hi=float(c.get("log_hi", 0.0)),
                )
                for c in g["clusters"]
            ],
        )
        for g in s["groups"]
    ]
    return RankWindowSummary(
        job_id=s["job_id"],
        rank=int(s["rank"]),
        window_id=s["window_id"],
        groups=groups,
        raw_event_bytes_est=int(s.get("raw_event_bytes_est", 0)),
        summary_bytes_est=int(s.get("summary_bytes_est", 0)),
    )


def run_synthetic_job(
    *,
    data_dir: str,
    job_id: str = "synth",
    n_ranks: int = 8,
    n_steps: int = 40,
    straggler: int = 5,
    onset: int = 15,
    slow_factor: float = 2.8,
    seed: int = 0,
) -> dict[str, Any]:
    """Simulate DP ranks with a compute straggler; run Producer→Processor→L1–L3."""
    rng = np.random.default_rng(seed)
    metrics = MetricStore(data_dir + "/metrics")
    objects = ObjectStore(data_dir + "/objects")
    processor = Processor(metrics=metrics, objects=objects, job_id=job_id)

    summary_dicts: list[dict[str, Any]] = []
    all_iterations: list[IterationSample] = []
    all_phases: list[PhaseEvent] = []

    for rank in range(n_ranks):
        producer = TraceProducer(rank=rank, job_id=job_id, use_cupti=False)
        producer.begin_window()
        for step in range(n_steps):
            attn = 6.0 + 0.4 * rng.normal()
            mlp = 5.0 + 0.3 * rng.normal()
            ar = 4.0 + 0.2 * rng.normal()
            if rank == straggler and step >= onset:
                attn *= slow_factor * 8.0
                mlp *= slow_factor * 7.0
            attn = max(attn, 0.1)
            mlp = max(mlp, 0.1)
            ar = max(ar, 0.1)
            step_ms = attn + mlp + ar

            producer.semantics.events.append(
                PhaseEvent("self_attention", attn, rank=rank, step=step, group="dp")
            )
            producer.semantics.events.append(
                PhaseEvent("mlp", mlp, rank=rank, step=step, group="dp")
            )
            producer.semantics.events.append(
                PhaseEvent("allreduce", ar, rank=rank, step=step, group="dp")
            )
            producer.record_iteration(step, step_ms)

            for _ in range(3):
                g1 = 0.42 * (slow_factor if rank == straggler and step >= onset else 1.0)
                g1 *= float(rng.lognormal(0.0, 0.05))
                producer.emit_fake_kernel("gemm_fc1", 0, max(g1, 0.01))
                producer.emit_fake_kernel(
                    "gelu", 0, max(0.08 * float(rng.lognormal(0.0, 0.05)), 0.01)
                )
                g2 = 0.38 * (slow_factor if rank == straggler and step >= onset else 1.0)
                g2 *= float(rng.lognormal(0.0, 0.05))
                producer.emit_fake_kernel("gemm_fc2", 0, max(g2, 0.01))
                producer.emit_fake_kernel(
                    "layernorm", 1, max(0.05 * float(rng.lognormal(0.0, 0.05)), 0.01)
                )

        payload = producer.end_window()
        result = processor.ingest_payload(payload)
        summary_dicts.append(result["summary"])
        all_iterations.extend(
            IterationSample(
                rank=rank,
                step=int(it["step"]),
                duration_ms=float(it["duration_ms"]),
            )
            for it in payload["iterations"]
        )
        all_phases.extend(
            PhaseEvent(
                phase=p["phase"],
                duration_ms=float(p["duration_ms"]),
                rank=rank,
                step=int(p["step"]),
                group=str(p.get("group", "dp")),
            )
            for p in payload["phases"]
        )

    rank_summaries = [_summary_from_dict(s) for s in summary_dicts]
    report = diagnose(
        job_id=job_id,
        window_id="w0",
        iterations=all_iterations,
        phases=all_phases,
        summaries=rank_summaries,
        l3_targets=[("gemm_fc1", 0)],
    )
    metrics.write_alert(report.to_dict())
    mean_ratio = (
        float(np.mean([s.compression_ratio for s in rank_summaries]))
        if rank_summaries
        else 0.0
    )

    return {
        "job_id": job_id,
        "n_ranks": n_ranks,
        "n_steps": n_steps,
        "injected_straggler": straggler,
        "onset": onset,
        "slow_factor": slow_factor,
        "mean_compression_ratio": mean_ratio,
        "flagged_ranks": report.flagged_ranks,
        "detected": straggler in report.flagged_ranks,
        "notes": report.notes,
        "l1_global": report.l1.get("global", {}),
        "l2_flagged": report.l2.get("flagged_ranks", []),
        "l3_flagged": report.l3.get("flagged_ranks", []),
        "report": report.to_dict(),
        "data_dir": data_dir,
        "n_perfetto_objects": len(objects.list_keys("traces/")),
    }
