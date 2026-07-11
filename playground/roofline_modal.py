"""Roofline measurements on a Modal GPU for the roofline blog post.

Benchmarks a set of common DL ops (elementwise, norms, softmax, attention,
GEMMs at several shapes) in BF16, computes analytic FLOPs and bytes for each,
and reports achieved throughput, arithmetic intensity, and MFU / MBU against
both datasheet peaks and *measured* achievable peaks.

Usage (from repo root)::

    uv run modal run playground/roofline_modal.py
    uv run modal run playground/roofline_modal.py::decode
    uv run modal run playground/roofline_modal.py::train
    uv run modal run playground/roofline_modal.py::cv
    ROOFLINE_GPU=H100 uv run modal run playground/roofline_modal.py

Writes JSON under ``playground/``. Figures via ``roofline_figures.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# Datasheet peaks: dense (no sparsity) BF16 TFLOPS, HBM/GDDR bandwidth TB/s.
# Keyed by torch.cuda.get_device_name() — Modal's "A10G" slot can hand out a
# plain A10, and the two have very different BF16 peaks (125 vs 70 TFLOPS).
DATASHEET = {
    "NVIDIA A10G": {"peak_bf16_tflops": 70.0, "peak_mem_tbs": 0.6},
    "NVIDIA A10": {"peak_bf16_tflops": 125.0, "peak_mem_tbs": 0.6},
    "NVIDIA A100": {"peak_bf16_tflops": 312.0, "peak_mem_tbs": 2.039},
    "NVIDIA H100": {"peak_bf16_tflops": 989.0, "peak_mem_tbs": 3.35},
}

GPU = os.environ.get("ROOFLINE_GPU", "A10G")

app = modal.App("roofline")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "torchvision")


@app.function(gpu=GPU, image=image, timeout=1200)
def bench() -> dict:
    import torch
    import torch.nn.functional as F

    assert torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0)
    # Datasheet BF16 peaks assume FP32 accumulation. cuBLAS defaults to
    # reduced-precision BF16 reduction, which can exceed the datasheet number
    # on some parts and make "MFU" > 100%. Disable for apples-to-apples.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def time_op(fn, warmup: int = 10, iters: int = 50) -> float:
        """Median wall time of fn() in seconds, measured with CUDA events."""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3)
        times.sort()
        return times[len(times) // 2]

    results: dict = {"device": device_name, "ops": []}

    def record(name, kind, seconds, flops, bytes_moved, note=""):
        results["ops"].append(
            {
                "name": name,
                "kind": kind,
                "seconds": seconds,
                "flops": flops,
                "bytes": bytes_moved,
                "intensity": flops / bytes_moved,
                "tflops": flops / seconds / 1e12,
                "gbs": bytes_moved / seconds / 1e9,
                "note": note,
            }
        )

    dt = torch.bfloat16
    esize = 2  # bytes per bf16 element

    # ── Achievable ceilings ─────────────────────────────────────────────
    # Bandwidth: a big device-to-device copy reads N bytes and writes N bytes.
    n = 1 << 29  # 512 Mi elements = 1 GiB per tensor
    x = torch.randn(n, dtype=dt, device="cuda")
    y = torch.empty_like(x)
    s = time_op(lambda x=x, y=y: y.copy_(x))
    record("memcpy 1GiB (ceiling)", "ceiling_bw", s, 0, 2 * n * esize)
    del x, y

    # Compute: a large square GEMM is as close to peak BF16 as we get.
    n = 8192
    a = torch.randn(n, n, dtype=dt, device="cuda")
    b = torch.randn(n, n, dtype=dt, device="cuda")
    s = time_op(lambda a=a, b=b: a @ b)
    record("GEMM 8192^3 (ceiling)", "ceiling_flops", s, 2 * n**3, 3 * n * n * esize)
    del a, b

    # ── Memory-bound suspects ───────────────────────────────────────────
    n = 1 << 28  # 256 Mi elements
    x = torch.randn(n, dtype=dt, device="cuda")
    y = torch.randn(n, dtype=dt, device="cuda")

    # add: 1 FLOP/elem, read 2 elems write 1
    s = time_op(lambda x=x, y=y: torch.add(x, y))
    record("Elementwise add", "op", s, n, 3 * n * esize)

    # GELU (tanh approx): count ~10 FLOP/elem, read 1 write 1
    s = time_op(lambda x=x: F.gelu(x, approximate="tanh"))
    record("GELU", "op", s, 10 * n, 2 * n * esize)
    del x, y

    # RMSNorm / softmax over (8192, 8192)
    rows, cols = 8192, 8192
    x = torch.randn(rows, cols, dtype=dt, device="cuda")
    w = torch.ones(cols, dtype=dt, device="cuda")
    n = rows * cols
    s = time_op(lambda x=x, w=w: F.rms_norm(x, (cols,), weight=w))
    record("RMSNorm 8192x8192", "op", s, 4 * n, 2 * n * esize)
    s = time_op(lambda x=x: F.softmax(x, dim=-1))
    record("Softmax 8192x8192", "op", s, 5 * n, 2 * n * esize)
    del x, w

    # ── GEMM sweep: square N×N×N ────────────────────────────────────────
    for n in [256, 512, 1024, 2048, 4096, 8192]:
        a = torch.randn(n, n, dtype=dt, device="cuda")
        b = torch.randn(n, n, dtype=dt, device="cuda")
        s = time_op(lambda a=a, b=b: a @ b)
        record(f"GEMM {n}^3", "gemm_sweep", s, 2 * n**3, 3 * n * n * esize)
        del a, b

    # Decode-shaped GEMV: M=1, K=N=8192 — same "GEMM", wildly different I.
    k = 8192
    a = torch.randn(1, k, dtype=dt, device="cuda")
    b = torch.randn(k, k, dtype=dt, device="cuda")
    s = time_op(lambda a=a, b=b: a @ b)
    record("GEMM M=1 K=N=8192 (decode)", "op", s, 2 * k * k, (k * k + 2 * k) * esize)
    del a, b

    # ── Attention: math backend vs flash backend ────────────────────────
    bsz, heads, seq, hd = 4, 16, 4096, 64
    q = torch.randn(bsz, heads, seq, hd, dtype=dt, device="cuda")
    k_ = torch.randn_like(q)
    v = torch.randn_like(q)
    att_flops = 4 * bsz * heads * seq * seq * hd  # QK^T + PV
    io_qkvo = 4 * bsz * heads * seq * hd * esize
    # Vanilla materializes the S×S score matrix in HBM: written by QK^T,
    # read+written by softmax, read by PV → ~4 traversals of S² elements.
    scores_traffic = 4 * bsz * heads * seq * seq * esize

    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        s = time_op(lambda: F.scaled_dot_product_attention(q, k_, v), warmup=3, iters=10)
    record(
        "Attention vanilla S=4096",
        "op",
        s,
        att_flops,
        io_qkvo + scores_traffic,
        note="bytes assume ~4 HBM traversals of the S^2 score matrix",
    )
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        s = time_op(lambda: F.scaled_dot_product_attention(q, k_, v))
    record("FlashAttention S=4096", "op", s, att_flops, io_qkvo)

    return results


@app.function(gpu=GPU, image=image, timeout=1200)
def decode_sweep() -> dict:
    """Model-level roofline: decode-step throughput vs batch size.

    The "model" is an MLP-only LLM stand-in (~1.4B params). One decode step
    reads every weight once, so arithmetic intensity ~= batch size and the
    whole model traces out a roofline as batch grows.
    """
    import torch
    import torch.nn as nn

    assert torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    hidden, ffn, layers = 4096, 11008, 16
    blocks = []
    for _ in range(layers):
        blocks += [
            nn.Linear(hidden, ffn, bias=False),
            nn.GELU(),
            nn.Linear(ffn, hidden, bias=False),
        ]
    model = nn.Sequential(*blocks).to("cuda", torch.bfloat16).eval()
    n_params = sum(p.numel() for p in model.parameters())

    def time_step(x, warmup=5, iters=30):
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            torch.cuda.synchronize()
            times = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) / 1e3)
        times.sort()
        return times[len(times) // 2]

    points = []
    for bsz in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        x = torch.randn(bsz, hidden, dtype=torch.bfloat16, device="cuda")
        s = time_step(x)
        flops = 2 * n_params * bsz
        points.append(
            {
                "batch": bsz,
                "seconds": s,
                "tokens_per_s": bsz / s,
                "tflops": flops / s / 1e12,
            }
        )
        del x

    return {"device": device_name, "n_params": n_params, "points": points}


@app.function(gpu=GPU, image=image, timeout=1800)
def train_sweep() -> dict:
    """Model-level roofline for training: fwd+bwd throughput vs tokens/step.

    Same MLP stand-in as decode_sweep, but each step is forward + backward.
    FLOPs = 6*P*T; weight-related traffic is ~6P bytes (read weights in fwd,
    read again in bwd, write grads), so arithmetic intensity ~= T.
    """
    import torch
    import torch.nn as nn

    assert torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    hidden, ffn, layers = 4096, 11008, 16
    blocks = []
    for _ in range(layers):
        blocks += [
            nn.Linear(hidden, ffn, bias=False),
            nn.GELU(),
            nn.Linear(ffn, hidden, bias=False),
        ]
    model = nn.Sequential(*blocks).to("cuda", torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())

    def step(x):
        model.zero_grad(set_to_none=True)
        model(x).sum().backward()

    def time_step(x, warmup, iters):
        for _ in range(warmup):
            step(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3)
        times.sort()
        return times[len(times) // 2]

    points = []
    for toks in [1, 4, 16, 64, 128, 256, 512, 1024, 4096]:
        x = torch.randn(toks, hidden, dtype=torch.bfloat16, device="cuda")
        iters = 20 if toks <= 512 else 8
        s = time_step(x, warmup=3, iters=iters)
        flops = 6 * n_params * toks
        points.append(
            {
                "tokens": toks,
                "seconds": s,
                "tokens_per_s": toks / s,
                "tflops": flops / s / 1e12,
            }
        )
        del x

    return {"device": device_name, "n_params": n_params, "points": points}


def _tensor_nbytes(t) -> int:
    import torch

    if torch.is_tensor(t):
        return t.numel() * t.element_size()
    if isinstance(t, (tuple, list)):
        return sum(_tensor_nbytes(x) for x in t)
    if isinstance(t, dict):
        return sum(_tensor_nbytes(x) for x in t.values())
    return 0


def _count_flops_and_bytes(model, run_fn) -> tuple[int, int]:
    """General (non-6PT) accounting: FlopCounterMode + leaf-module IO bytes.

    Bytes are a *module-boundary* estimate: each leaf module charges its
    parameter reads plus input/output tensor sizes. That over-counts when
    activations stay in cache / get fused, and under-counts workspace not
    visible at module boundaries. It is still the right *portable* method
    when you do not have ncu DRAM metrics — and it works for any nn.Module.
    """
    from torch.utils.flop_counter import FlopCounterMode

    bytes_acc = {"n": 0}

    def pre_hook(mod, inputs):
        for p in mod.parameters(recurse=False):
            bytes_acc["n"] += p.numel() * p.element_size()
        bytes_acc["n"] += _tensor_nbytes(inputs)

    def post_hook(mod, inputs, output):
        bytes_acc["n"] += _tensor_nbytes(output)

    handles = []
    for mod in model.modules():
        if any(mod.children()):
            continue
        handles.append(mod.register_forward_pre_hook(pre_hook))
        handles.append(mod.register_forward_hook(post_hook))

    with FlopCounterMode(display=False) as fcm:
        run_fn()
    flops = int(fcm.get_total_flops())

    for h in handles:
        h.remove()
    return flops, int(bytes_acc["n"])


@app.function(gpu=GPU, image=image, timeout=1800)
def cv_sweep() -> dict:
    """MFU/MBU for real CV models without LLM 6PT formulas.

    FLOPs: torch.utils.flop_counter.FlopCounterMode (fwd, or fwd+bwd)
    Bytes: leaf-module parameter + activation IO estimate (see helper)
    Time: median CUDA events
    """
    import torch
    import torchvision.models as tvm

    assert torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = True

    builders = {
        "resnet50": lambda: tvm.resnet50(weights=None),
        "vit_b_16": lambda: tvm.vit_b_16(weights=None),
    }

    def time_fn(fn, warmup=5, iters=20) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3)
        times.sort()
        return times[len(times) // 2]

    models_out = []
    for name, build in builders.items():
        # FP32 master weights + BF16 autocast: BN stays stable, conv/matmul hit TC.
        model = build().to("cuda")
        n_params = sum(p.numel() for p in model.parameters())
        entry = {"name": name, "n_params": n_params, "infer": [], "train": []}

        for phase in ("infer", "train"):
            points = []
            for bsz in [1, 2, 4, 8, 16, 32, 64]:
                x = torch.randn(bsz, 3, 224, 224, device="cuda", dtype=torch.float32)
                if phase == "infer":
                    model.eval()

                    def run(x=x, model=model):
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            model(x)

                    flops, nbytes = _count_flops_and_bytes(model, run)
                    s = time_fn(run, warmup=5, iters=30 if bsz <= 16 else 12)
                else:
                    model.train()

                    def run(x=x, model=model):
                        model.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = model(x).sum()
                        loss.backward()

                    flops, nbytes = _count_flops_and_bytes(model, run)
                    # Forward hooks miss bwd weight re-read + grad write.
                    # Params are FP32 master (4 bytes); grads match that storage.
                    nbytes = nbytes + 8 * n_params
                    s = time_fn(run, warmup=3, iters=12 if bsz <= 16 else 6)

                points.append(
                    {
                        "batch": bsz,
                        "seconds": s,
                        "flops": flops,
                        "bytes": nbytes,
                        "intensity": flops / nbytes if nbytes else 0.0,
                        "tflops": flops / s / 1e12,
                        "gbs": nbytes / s / 1e9,
                        "images_per_s": bsz / s,
                    }
                )
                del x
            entry[phase] = points
        models_out.append(entry)
        del model
        torch.cuda.empty_cache()

    return {"device": device_name, "models": models_out}


def _attach_datasheet(results: dict) -> None:
    results["gpu"] = GPU
    key = max((k for k in DATASHEET if results["device"].startswith(k)), key=len)
    results.update(DATASHEET[key])


@app.local_entrypoint()
def main():
    """Per-op roofline benchmark. Run: modal run playground/roofline_modal.py"""
    results = bench.remote()
    _attach_datasheet(results)
    out = Path(__file__).parent / f"roofline_{GPU.lower().replace('-', '_')}_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")
    for op in results["ops"]:
        print(
            f"{op['name']:32s} I={op['intensity']:9.2f}  "
            f"{op['tflops']:8.2f} TFLOP/s  {op['gbs']:8.1f} GB/s"
        )


@app.local_entrypoint()
def decode():
    """Model-level decode sweep. Run: modal run playground/roofline_modal.py::decode"""
    results = decode_sweep.remote()
    _attach_datasheet(results)
    out = Path(__file__).parent / "roofline_decode_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}  ({results['device']}, P={results['n_params'] / 1e9:.2f}B)")
    for p in results["points"]:
        print(
            f"batch={p['batch']:4d}  {p['tokens_per_s']:10.1f} tok/s  "
            f"{p['tflops']:8.2f} TFLOP/s  {p['seconds'] * 1e3:7.2f} ms"
        )


@app.local_entrypoint()
def train():
    """Model-level training sweep. Run: modal run playground/roofline_modal.py::train"""
    results = train_sweep.remote()
    _attach_datasheet(results)
    out = Path(__file__).parent / "roofline_train_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}  ({results['device']}, P={results['n_params'] / 1e9:.2f}B)")
    for p in results["points"]:
        print(
            f"tokens={p['tokens']:5d}  {p['tokens_per_s']:10.1f} tok/s  "
            f"{p['tflops']:8.2f} TFLOP/s  {p['seconds'] * 1e3:7.2f} ms"
        )


@app.local_entrypoint()
def cv():
    """CV models with instrumented MFU/MBU. Run: modal run playground/roofline_modal.py::cv"""
    results = cv_sweep.remote()
    _attach_datasheet(results)
    peak_f = results["peak_bf16_tflops"]
    peak_b = results["peak_mem_tbs"] * 1000  # GB/s
    out = Path(__file__).parent / "roofline_cv_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}  ({results['device']})")
    for m in results["models"]:
        print(f"\n== {m['name']}  P={m['n_params'] / 1e6:.1f}M ==")
        for phase in ("infer", "train"):
            print(f"  [{phase}]")
            for p in m[phase]:
                mfu = 100 * p["tflops"] / peak_f
                mbu = 100 * p["gbs"] / peak_b
                print(
                    f"    B={p['batch']:3d}  {p['images_per_s']:7.1f} img/s  "
                    f"I={p['intensity']:6.1f}  MFU={mfu:5.1f}%  MBU={mbu:5.1f}%  "
                    f"{p['tflops']:5.2f} TFLOP/s  {p['gbs']:6.1f} GB/s"
                )
