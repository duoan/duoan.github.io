"""CPU call-stack sampling channel (§4.1).

Production ARGUS uses streaming py-spy. Here we provide:
* ``PySpyStackSampler`` — thin wrapper when ``py-spy`` is on PATH
* ``FakeStackSampler`` — deterministic host-stall frames for demos
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any

from argus.schemas import StackSnapshot


class FakeStackSampler:
    def __init__(self, rank: int = 0):
        self.rank = rank
        self._snaps: list[StackSnapshot] = []
        self._running = False

    def start(self) -> None:
        self._running = True
        self._snaps.clear()

    def stop(self) -> None:
        self._running = False

    def note_stall(self, frames: list[str] | None = None) -> None:
        if not self._running:
            return
        self._snaps.append(
            StackSnapshot(
                timestamp_ns=time.time_ns(),
                frames=frames
                or [
                    "torch._inductor.compile",
                    "flash_attn_jit",
                    "site-packages/torch/...",
                ],
                rank=self.rank,
            )
        )

    def snapshots(self) -> list[StackSnapshot]:
        return list(self._snaps)


class PySpyStackSampler:
    """Best-effort py-spy dump for a PID (optional; not required for Modal demos)."""

    def __init__(self, pid: int, rank: int = 0):
        self.pid = pid
        self.rank = rank
        self._snaps: list[StackSnapshot] = []
        if shutil.which("py-spy") is None:
            raise RuntimeError("py-spy not found on PATH")

    def sample_once(self) -> StackSnapshot:
        proc = subprocess.run(
            ["py-spy", "dump", "--pid", str(self.pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        frames = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        snap = StackSnapshot(
            timestamp_ns=time.time_ns(), frames=frames[:64], rank=self.rank
        )
        self._snaps.append(snap)
        return snap

    def snapshots(self) -> list[StackSnapshot]:
        return list(self._snaps)

    def info(self) -> dict[str, Any]:
        return {"backend": "py-spy", "pid": self.pid, "n_snapshots": len(self._snaps)}
