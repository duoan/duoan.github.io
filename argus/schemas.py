"""Shared event / summary schemas for the ARGUS pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class KernelEvent:
    """One GPU kernel observation (CUPTI Activity API style)."""

    name: str
    stream: int
    duration_ms: float
    start_ns: int = 0
    rank: int = 0


@dataclass(slots=True)
class PhaseEvent:
    """Framework semantics phase duration (CUDA Event style)."""

    phase: str
    duration_ms: float
    rank: int = 0
    step: int = 0
    group: str = "dp"  # parallelism group tag for L2 routing


@dataclass(slots=True)
class StackSnapshot:
    """Streaming CPU call-stack sample (py-spy style)."""

    timestamp_ns: int
    frames: list[str]
    rank: int = 0


@dataclass(slots=True)
class IterationSample:
    rank: int
    step: int
    duration_ms: float


@dataclass(slots=True)
class ClusterStats:
    count: int
    p50_ms: float
    p99_ms: float
    log_lo: float = 0.0
    log_hi: float = 0.0


@dataclass(slots=True)
class KernelGroupSummary:
    kernel: str
    stream: int
    samples: int
    clusters: list[ClusterStats]


@dataclass(slots=True)
class RankWindowSummary:
    """Compressed per-rank kernel window (Metric Storage payload)."""

    job_id: str
    rank: int
    window_id: str
    groups: list[KernelGroupSummary]
    raw_event_bytes_est: int = 0
    summary_bytes_est: int = 0

    @property
    def compression_ratio(self) -> float:
        return self.raw_event_bytes_est / max(self.summary_bytes_est, 1)


@dataclass(slots=True)
class DiagnosisReport:
    job_id: str
    window_id: str
    l1: dict[str, Any] = field(default_factory=dict)
    l2: dict[str, Any] = field(default_factory=dict)
    l3: dict[str, Any] = field(default_factory=dict)
    flagged_ranks: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summary_to_dict(summary: RankWindowSummary) -> dict[str, Any]:
    return asdict(summary)
