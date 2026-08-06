"""File-backed metric storage (Prometheus Remote Write stand-in)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from argus.schemas import IterationSample, PhaseEvent, RankWindowSummary, summary_to_dict


class MetricStore:
    """Append-only JSONL metrics under a data directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, kind: str) -> Path:
        return self.root / f"{kind}.jsonl"

    def _append(self, kind: str, row: dict[str, Any]) -> None:
        with self._lock, self._path(kind).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def write_iteration(self, sample: IterationSample, *, job_id: str) -> None:
        self._append(
            "iterations",
            {
                "job_id": job_id,
                "rank": sample.rank,
                "step": sample.step,
                "duration_ms": sample.duration_ms,
            },
        )

    def write_phase(self, phase: PhaseEvent, *, job_id: str) -> None:
        self._append(
            "phases",
            {
                "job_id": job_id,
                "rank": phase.rank,
                "step": phase.step,
                "phase": phase.phase,
                "duration_ms": phase.duration_ms,
                "group": phase.group,
            },
        )

    def write_kernel_summary(self, summary: RankWindowSummary) -> None:
        self._append("kernel_summaries", summary_to_dict(summary))

    def write_alert(self, alert: dict[str, Any]) -> None:
        self._append("alerts", alert)

    def load_jsonl(self, kind: str) -> list[dict[str, Any]]:
        path = self._path(kind)
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
