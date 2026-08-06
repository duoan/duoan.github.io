"""FT-Client-lite: run diagnosis against a MetricStore directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from argus.analysis.diagnose import diagnose
from argus.schemas import (
    ClusterStats,
    IterationSample,
    KernelGroupSummary,
    PhaseEvent,
    RankWindowSummary,
)
from argus.storage.metrics import MetricStore


def _load_summaries(rows: list[dict]) -> list[RankWindowSummary]:
    out: list[RankWindowSummary] = []
    for r in rows:
        groups = []
        for g in r.get("groups", []):
            clusters = [
                ClusterStats(
                    count=int(c["count"]),
                    p50_ms=float(c["p50_ms"]),
                    p99_ms=float(c["p99_ms"]),
                    log_lo=float(c.get("log_lo", 0.0)),
                    log_hi=float(c.get("log_hi", 0.0)),
                )
                for c in g.get("clusters", [])
            ]
            groups.append(
                KernelGroupSummary(
                    kernel=g["kernel"],
                    stream=int(g["stream"]),
                    samples=int(g["samples"]),
                    clusters=clusters,
                )
            )
        out.append(
            RankWindowSummary(
                job_id=r["job_id"],
                rank=int(r["rank"]),
                window_id=r["window_id"],
                groups=groups,
                raw_event_bytes_est=int(r.get("raw_event_bytes_est", 0)),
                summary_bytes_est=int(r.get("summary_bytes_est", 0)),
            )
        )
    return out


def run_client(data_dir: Path, job_id: str = "default") -> dict:
    store = MetricStore(data_dir)
    iterations = [
        IterationSample(
            rank=int(r["rank"]),
            step=int(r["step"]),
            duration_ms=float(r["duration_ms"]),
        )
        for r in store.load_jsonl("iterations")
        if r.get("job_id", job_id) == job_id
    ]
    phases = [
        PhaseEvent(
            phase=r["phase"],
            duration_ms=float(r["duration_ms"]),
            rank=int(r["rank"]),
            step=int(r["step"]),
            group=str(r.get("group", "dp")),
        )
        for r in store.load_jsonl("phases")
        if r.get("job_id", job_id) == job_id
    ]
    summaries = [
        s
        for s in _load_summaries(store.load_jsonl("kernel_summaries"))
        if s.job_id == job_id
    ]
    window_id = summaries[0].window_id if summaries else "latest"
    report = diagnose(
        job_id=job_id,
        window_id=window_id,
        iterations=iterations,
        phases=phases,
        summaries=summaries,
    )
    store.write_alert(report.to_dict())
    return report.to_dict()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="ARGUS FT-Client-lite")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--job-id", default="default")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    report = run_client(args.data_dir, job_id=args.job_id)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)
    print("flagged_ranks=", report["flagged_ranks"])
    print("notes=", report["notes"])


if __name__ == "__main__":
    main()
