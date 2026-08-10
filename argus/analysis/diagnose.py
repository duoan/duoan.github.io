"""Progressive diagnosis orchestrator (L1 ∥ L2 ∥ L3)."""

from __future__ import annotations

from typing import Any

from argus.analysis.l1 import run_l1
from argus.analysis.l2 import run_l2
from argus.analysis.l3 import run_l3
from argus.schemas import (
    DiagnosisReport,
    IterationSample,
    PhaseEvent,
    RankWindowSummary,
)


def diagnose(
    *,
    job_id: str,
    window_id: str,
    iterations: list[IterationSample],
    phases: list[PhaseEvent],
    summaries: list[RankWindowSummary],
    l3_targets: list[tuple[str, int]] | None = None,
) -> DiagnosisReport:
    l1 = run_l1(iterations)
    l2 = run_l2(phases)
    l3 = run_l3(summaries, target_kernels=l3_targets)

    flagged = sorted(
        set(l1.get("anomalous_ranks", []))
        | set(l2.get("flagged_ranks", []))
        | set(l3.get("flagged_ranks", []))
    )
    notes: list[str] = []
    if l1["global"]["label"] != "stable":
        notes.append(f"L1 global iteration={l1['global']['label']}")
    if l2["flagged_ranks"]:
        notes.append(f"L2 stragglers={l2['flagged_ranks']}")
    if l3["flagged_ranks"]:
        notes.append(f"L3 kernel outliers={l3['flagged_ranks']}")
    if not notes:
        notes.append("No automated anomalies (stable)")

    return DiagnosisReport(
        job_id=job_id,
        window_id=window_id,
        l1=l1,
        l2=l2,
        l3=l3,
        flagged_ranks=flagged,
        notes=notes,
    )


def report_to_dict(report: DiagnosisReport) -> dict[str, Any]:
    return report.to_dict()
