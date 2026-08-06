"""Framework semantics instrumentation via CUDA Events (§4.2)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from argus.schemas import PhaseEvent


class SemanticsTracer:
    """Record GPU-side phase durations with CUDA Events when available."""

    def __init__(self, rank: int = 0):
        self.rank = rank
        self.events: list[PhaseEvent] = []
        self._step = 0

    def set_step(self, step: int) -> None:
        self._step = step

    @contextmanager
    def phase(self, name: str, *, group: str = "dp") -> Iterator[None]:
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]

        if torch is None or not torch.cuda.is_available():
            # CPU fallback: wall clock (not GPU-accurate, fine for synth demos).
            import time

            t0 = time.perf_counter()
            yield
            dur = (time.perf_counter() - t0) * 1e3
            self.events.append(
                PhaseEvent(
                    phase=name,
                    duration_ms=dur,
                    rank=self.rank,
                    step=self._step,
                    group=group,
                )
            )
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        end.synchronize()
        self.events.append(
            PhaseEvent(
                phase=name,
                duration_ms=float(start.elapsed_time(end)),
                rank=self.rank,
                step=self._step,
                group=group,
            )
        )

    def pop_events(self) -> list[PhaseEvent]:
        out = self.events
        self.events = []
        return out
