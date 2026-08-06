"""Minimal Perfetto JSON exporter for deep-dive (L4) traces.

Emits Chrome Trace Event Format JSON that Perfetto UI can load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from argus.schemas import KernelEvent, PhaseEvent, StackSnapshot


def events_to_perfetto(
    *,
    kernels: list[KernelEvent],
    phases: list[PhaseEvent] | None = None,
    stacks: list[StackSnapshot] | None = None,
    rank: int = 0,
) -> dict[str, Any]:
    trace_events: list[dict[str, Any]] = []
    pid = rank + 1

    # Kernel track
    for i, k in enumerate(kernels):
        start_us = (k.start_ns / 1e3) if k.start_ns else float(i) * 1000.0
        dur_us = max(k.duration_ms * 1e3, 0.1)
        trace_events.append(
            {
                "name": k.name,
                "cat": "gpu",
                "ph": "X",
                "ts": start_us,
                "dur": dur_us,
                "pid": pid,
                "tid": 1000 + k.stream,
                "args": {"stream": k.stream, "duration_ms": k.duration_ms},
            }
        )

    # Phase track
    t = 0.0
    for p in phases or []:
        dur_us = max(p.duration_ms * 1e3, 0.1)
        trace_events.append(
            {
                "name": p.phase,
                "cat": "semantics",
                "ph": "X",
                "ts": t,
                "dur": dur_us,
                "pid": pid,
                "tid": 2000,
                "args": {"step": p.step, "group": p.group},
            }
        )
        t += dur_us

    # Stack samples as instant events
    for s in stacks or []:
        top = s.frames[0] if s.frames else "<empty>"
        trace_events.append(
            {
                "name": top,
                "cat": "cpu",
                "ph": "i",
                "ts": s.timestamp_ns / 1e3,
                "pid": pid,
                "tid": 3000,
                "s": "t",
                "args": {"frames": s.frames[:16]},
            }
        )

    return {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
        "otherData": {"rank": rank, "source": "argus"},
    }


def write_perfetto(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path
