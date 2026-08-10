"""Trace Producer agent: bundles three observation channels (§4)."""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from argus.producer.cupti import CuptiTracer, FakeKernelTracer
from argus.producer.semantics import SemanticsTracer
from argus.producer.stacks import FakeStackSampler
from argus.schemas import IterationSample, KernelEvent


class TraceProducer:
    """Always-on producer sidecar API used by training loops / Modal demos."""

    def __init__(
        self,
        *,
        rank: int = 0,
        job_id: str = "default",
        use_cupti: bool | None = None,
        socket_path: str | Path | None = None,
    ):
        self.rank = rank
        self.job_id = job_id
        self.semantics = SemanticsTracer(rank=rank)
        self.stacks = FakeStackSampler(rank=rank)
        if use_cupti is None:
            use_cupti = os.environ.get("ARGUS_USE_CUPTI", "0") == "1"
        self.use_cupti = use_cupti
        self.kernels: CuptiTracer | FakeKernelTracer
        if use_cupti:
            self.kernels = CuptiTracer()
            self.kernels.rank = rank
        else:
            self.kernels = FakeKernelTracer(rank=rank)
        self.socket_path = Path(socket_path) if socket_path else None
        self._iterations: list[IterationSample] = []
        self._window = 0

    def begin_window(self) -> None:
        self.kernels.clear()
        self.kernels.start()
        self.stacks.start()
        self.semantics.events.clear()
        self._iterations.clear()

    def end_window(self) -> dict[str, Any]:
        self.kernels.stop()
        self.stacks.stop()
        window_id = f"w{self._window}"
        self._window += 1
        payload = {
            "job_id": self.job_id,
            "rank": self.rank,
            "window_id": window_id,
            "kernels": [asdict(e) for e in self.kernels.records()],
            "phases": [asdict(e) for e in self.semantics.pop_events()],
            "stacks": [asdict(e) for e in self.stacks.snapshots()],
            "iterations": [asdict(e) for e in self._iterations],
        }
        if self.socket_path is not None:
            self._send(payload)
        return payload

    def record_iteration(self, step: int, duration_ms: float) -> None:
        self._iterations.append(
            IterationSample(rank=self.rank, step=step, duration_ms=duration_ms)
        )

    def emit_fake_kernel(self, name: str, stream: int, duration_ms: float) -> None:
        if isinstance(self.kernels, FakeKernelTracer):
            self.kernels.emit(name, stream, duration_ms)
        else:
            # CUPTI path collects real kernels; ignore synthetic emits.
            return

    def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.socket_path is not None
        data = (json.dumps(payload) + "\n").encode()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            # Retry briefly while Processor comes up.
            deadline = time.time() + 5.0
            while True:
                try:
                    sock.connect(str(self.socket_path))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.time() > deadline:
                        raise
                    time.sleep(0.05)
            sock.sendall(data)
            resp = b""
            while not resp.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    break
                resp += chunk
        return json.loads(resp.decode()) if resp.strip() else {}


def kernels_from_dicts(rows: list[dict[str, Any]], rank: int = 0) -> list[KernelEvent]:
    return [
        KernelEvent(
            name=r["name"],
            stream=int(r["stream"]),
            duration_ms=float(r["duration_ms"]),
            start_ns=int(r.get("start_ns", 0)),
            rank=rank,
        )
        for r in rows
    ]
