"""Processor pipeline: ingest producer payloads → compress → store (§5)."""

from __future__ import annotations

import json
import socketserver
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from argus.processor.compress import compress_kernel_events
from argus.processor.perfetto import events_to_perfetto
from argus.schemas import IterationSample, KernelEvent, PhaseEvent, StackSnapshot
from argus.storage.metrics import MetricStore
from argus.storage.objects import ObjectStore


class Processor:
    """In-process (and optional Unix-socket) Processor."""

    def __init__(
        self,
        *,
        metrics: MetricStore,
        objects: ObjectStore,
        job_id: str = "default",
    ):
        self.metrics = metrics
        self.objects = objects
        self.job_id = job_id
        self._sock_server: socketserver.ThreadingUnixStreamServer | None = None
        self._thread: threading.Thread | None = None

    def ingest_window(
        self,
        *,
        rank: int,
        window_id: str,
        kernels: list[KernelEvent],
        phases: list[PhaseEvent] | None = None,
        stacks: list[StackSnapshot] | None = None,
        iterations: list[IterationSample] | None = None,
    ) -> dict[str, Any]:
        phases = phases or []
        stacks = stacks or []
        iterations = iterations or []

        for it in iterations:
            self.metrics.write_iteration(it, job_id=self.job_id)
        for ph in phases:
            self.metrics.write_phase(ph, job_id=self.job_id)

        summary = compress_kernel_events(
            kernels, job_id=self.job_id, rank=rank, window_id=window_id
        )
        self.metrics.write_kernel_summary(summary)

        perfetto = events_to_perfetto(
            kernels=kernels, phases=phases, stacks=stacks, rank=rank
        )
        key = f"traces/{self.job_id}/rank{rank}/{window_id}.json"
        self.objects.put_json(key, perfetto)

        return {
            "rank": rank,
            "window_id": window_id,
            "compression_ratio": summary.compression_ratio,
            "raw_bytes": summary.raw_event_bytes_est,
            "summary_bytes": summary.summary_bytes_est,
            "n_groups": len(summary.groups),
            "perfetto_key": key,
            "summary": asdict(summary),
        }

    def ingest_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ingest a JSON producer payload."""
        rank = int(payload["rank"])
        window_id = str(payload.get("window_id", "w0"))
        kernels = [
            KernelEvent(
                name=e["name"],
                stream=int(e["stream"]),
                duration_ms=float(e["duration_ms"]),
                start_ns=int(e.get("start_ns", 0)),
                rank=rank,
            )
            for e in payload.get("kernels", [])
        ]
        phases = [
            PhaseEvent(
                phase=e["phase"],
                duration_ms=float(e["duration_ms"]),
                rank=rank,
                step=int(e.get("step", 0)),
                group=str(e.get("group", "dp")),
            )
            for e in payload.get("phases", [])
        ]
        stacks = [
            StackSnapshot(
                timestamp_ns=int(e["timestamp_ns"]),
                frames=list(e.get("frames", [])),
                rank=rank,
            )
            for e in payload.get("stacks", [])
        ]
        iterations = [
            IterationSample(
                rank=rank,
                step=int(e["step"]),
                duration_ms=float(e["duration_ms"]),
            )
            for e in payload.get("iterations", [])
        ]
        return self.ingest_window(
            rank=rank,
            window_id=window_id,
            kernels=kernels,
            phases=phases,
            stacks=stacks,
            iterations=iterations,
        )

    def start_unix_server(self, sock_path: Path) -> None:
        """Listen for newline-delimited JSON producer payloads."""
        sock_path = Path(sock_path)
        if sock_path.exists():
            sock_path.unlink()
        sock_path.parent.mkdir(parents=True, exist_ok=True)
        processor = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:  # noqa: N802
                for line in self.rfile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        result = processor.ingest_payload(payload)
                        self.wfile.write((json.dumps(result) + "\n").encode())
                    except Exception as exc:  # noqa: BLE001
                        self.wfile.write(
                            (json.dumps({"error": repr(exc)}) + "\n").encode()
                        )

        self._sock_server = socketserver.ThreadingUnixStreamServer(
            str(sock_path), Handler
        )
        self._thread = threading.Thread(
            target=self._sock_server.serve_forever, daemon=True
        )
        self._thread.start()

    def stop_unix_server(self) -> None:
        if self._sock_server is not None:
            self._sock_server.shutdown()
            self._sock_server = None
