"""CUPTI Activity API kernel tracer wrapper (§4.3)."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

from argus.schemas import KernelEvent

_NATIVE = Path(__file__).resolve().parent.parent / "native" / "cupti_tracer.cpp"


def build_tracer_so(out_path: Path, src: Path | None = None) -> dict:
    src = src or _NATIVE
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    include = f"{cuda_home}/include"
    lib = f"{cuda_home}/lib64"
    activity_h = Path(include) / "cupti_activity.h"
    if not activity_h.exists():
        activity_h = Path(include) / "cupti.h"
    probe = subprocess.check_output(
        [
            "bash",
            "-lc",
            f"grep -oE 'CUpti_ActivityKernel[0-9]+' {activity_h} | sort -V | uniq | tail -1",
        ],
        text=True,
    ).strip()
    kernel_t = probe or "CUpti_ActivityKernel4"
    cmd = [
        "g++",
        "-shared",
        "-fPIC",
        "-O2",
        f"-DARGUS_KERNEL_T={kernel_t}",
        f"-I{include}",
        str(src),
        f"-L{lib}",
        "-lcupti",
        "-Wl,-rpath," + lib,
        "-o",
        str(out_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"CUPTI tracer build failed ({proc.returncode}):\n"
            f"cmd={' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return {"kernel_struct": kernel_t, "so": str(out_path), "cmd": cmd}


class CuptiTracer:
    """ctypes wrapper around libargus_cupti_tracer.so."""

    def __init__(self, so_path: Path | None = None):
        if so_path is None:
            so_path = Path(os.environ.get("ARGUS_CUPTI_SO", "/tmp/libargus_cupti_tracer.so"))
        if not so_path.exists():
            build_tracer_so(so_path)
        self.so_path = so_path
        self.lib = ctypes.CDLL(str(so_path))
        self.lib.argus_cupti_start.restype = ctypes.c_int
        self.lib.argus_cupti_stop.restype = ctypes.c_int
        self.lib.argus_cupti_count.restype = ctypes.c_size_t
        self.lib.argus_cupti_clear.restype = None
        self.lib.argus_cupti_get.restype = ctypes.c_int
        self.lib.argus_cupti_kernel_struct.restype = ctypes.c_char_p
        self.rank = 0

    def start(self) -> None:
        if self.lib.argus_cupti_start() != 0:
            raise RuntimeError("argus_cupti_start failed")

    def stop(self) -> None:
        if self.lib.argus_cupti_stop() != 0:
            raise RuntimeError("argus_cupti_stop failed")

    def clear(self) -> None:
        self.lib.argus_cupti_clear()

    def kernel_struct(self) -> str:
        raw = self.lib.argus_cupti_kernel_struct()
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    def records(self) -> list[KernelEvent]:
        n = int(self.lib.argus_cupti_count())
        out: list[KernelEvent] = []
        name_buf = ctypes.create_string_buffer(128)
        stream = ctypes.c_uint32()
        dur = ctypes.c_double()
        for i in range(n):
            rc = self.lib.argus_cupti_get(
                ctypes.c_size_t(i),
                name_buf,
                ctypes.c_size_t(128),
                ctypes.byref(stream),
                ctypes.byref(dur),
            )
            if rc != 0:
                continue
            out.append(
                KernelEvent(
                    name=name_buf.value.decode("utf-8", "replace"),
                    stream=int(stream.value),
                    duration_ms=float(dur.value),
                    rank=self.rank,
                )
            )
        return out


class FakeKernelTracer:
    """Synthetic kernel events for CPU-only / Modal fallback tests."""

    def __init__(self, rank: int = 0):
        self.rank = rank
        self._buf: list[KernelEvent] = []

    def start(self) -> None:
        self._buf.clear()

    def stop(self) -> None:
        return

    def clear(self) -> None:
        self._buf.clear()

    def emit(self, name: str, stream: int, duration_ms: float) -> None:
        self._buf.append(
            KernelEvent(
                name=name, stream=stream, duration_ms=duration_ms, rank=self.rank
            )
        )

    def records(self) -> list[KernelEvent]:
        return list(self._buf)
