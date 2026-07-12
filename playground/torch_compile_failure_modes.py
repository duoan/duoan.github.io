"""Reproduce common torch.compile failure modes on CPU.

Demos:
  1. Graph break from data-dependent Python control flow
  2. Graph break from ``print`` / ``.item()`` (caught by ``fullgraph=True``)
  3. Recompile thrash on shape change (and how automatic dynamic mutes it)
  4. Recompile from a Python scalar (e.g. learning rate)

Usage (from repo root)::

    uv run python playground/torch_compile_failure_modes.py
"""

from __future__ import annotations

import torch
import torch._dynamo.config as dynamo_config
from torch._dynamo import reset
from torch._dynamo.testing import CompileCounter


def _banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def demo_graph_break_python_if() -> None:
    _banner("1. Graph break: data-dependent Python if")
    reset()
    x = torch.randn(8)

    def broken(x: torch.Tensor) -> torch.Tensor:
        # Pulls a tensor into Python control flow → Dynamo cannot keep one graph.
        if x.sum().item() > 0:
            return x * 2
        return x

    def fixed(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x.sum() > 0, x * 2, x)

    bad = torch._dynamo.explain(broken)(x)
    good = torch._dynamo.explain(fixed)(x)

    print(
        f"broken:  graphs={bad.graph_count}  "
        f"graph_breaks={bad.graph_break_count}  "
        f"break_reasons={len(bad.break_reasons)}"
    )
    if bad.break_reasons:
        print(f"  reason: {bad.break_reasons[0].reason}")
    print(
        f"fixed:   graphs={good.graph_count}  "
        f"graph_breaks={good.graph_break_count}  "
        f"break_reasons={len(good.break_reasons)}"
    )

    reset()
    try:
        torch.compile(broken, fullgraph=True, backend="eager")(x)
        print("fullgraph(broken): unexpectedly succeeded")
    except Exception as e:
        print(f"fullgraph(broken): {type(e).__name__}: {str(e).splitlines()[0][:140]}")

    reset()
    y = torch.compile(fixed, fullgraph=True, backend="eager")(x)
    print(f"fullgraph(fixed):   ok  out.mean={y.mean():.4f}")
    print("fix: keep control flow in tensor ops (torch.where / masks)")


def demo_graph_break_print() -> None:
    _banner("2. Graph break: print (silent until fullgraph=True)")
    reset()
    x = torch.randn(4)

    def with_print(x: torch.Tensor) -> torch.Tensor:
        print("debug:", float(x.mean()))
        return x.sin() + x.cos()

    # Without fullgraph, Dynamo breaks, runs eager for the side effect, continues.
    y = torch.compile(with_print, backend="eager")(x)
    print(f"compile without fullgraph: ok, out.mean={y.mean():.4f}")

    reset()
    try:
        torch.compile(with_print, fullgraph=True, backend="eager")(x)
        print("fullgraph unexpectedly succeeded")
    except Exception as e:
        print(f"fullgraph=True: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
    print("fix: strip side effects from the hot path, or compile a smaller region")


def demo_recompile_shapes() -> None:
    _banner("3. Recompile thrash: shapes (static vs automatic-dynamic vs dynamic)")
    shapes = [32, 32, 64, 128]

    def body(x: torch.Tensor) -> torch.Tensor:
        return (x * 2).relu()

    # Classic specialization: every new shape is a new compile.
    reset()
    static_counter = CompileCounter()
    with dynamo_config.patch(automatic_dynamic_shapes=False):
        static_fn = torch.compile(body, backend=static_counter)
        for n in shapes:
            static_fn(torch.randn(n))
            print(
                f"  automatic_dynamic=False  shape={n:>3}  "
                f"compiles={static_counter.frame_count}"
            )

    # Default in recent PyTorch: after one shape change, guards widen automatically.
    reset()
    auto_counter = CompileCounter()
    auto_fn = torch.compile(body, backend=auto_counter)
    for n in shapes:
        auto_fn(torch.randn(n))
        print(
            f"  automatic_dynamic=True   shape={n:>3}  "
            f"compiles={auto_counter.frame_count}"
        )

    reset()
    dyn_counter = CompileCounter()
    dyn_fn = torch.compile(body, backend=dyn_counter, dynamic=True)
    for n in shapes:
        dyn_fn(torch.randn(n))
        print(
            f"  dynamic=True             shape={n:>3}  "
            f"compiles={dyn_counter.frame_count}"
        )

    print(
        "summary: "
        f"strict-static={static_counter.frame_count}, "
        f"auto-dynamic={auto_counter.frame_count}, "
        f"dynamic=True={dyn_counter.frame_count}"
    )
    print(
        "fix: prefer intentional dynamic/bucketing; "
        "do not rely on automatic dynamic alone for production shape noise"
    )


def demo_recompile_scalar() -> None:
    _banner("4. Recompile: Python float specialized as a constant")
    reset()
    counter = CompileCounter()

    def scale(x: torch.Tensor, lr: float) -> torch.Tensor:
        return x * lr

    fn = torch.compile(scale, backend=counter)
    x = torch.randn(16)
    fn(x, 0.01)
    print(f"  lr=0.01           compiles={counter.frame_count}")
    fn(x, 0.01)
    print(f"  lr=0.01 again     compiles={counter.frame_count}  (cache hit)")
    fn(x, 0.02)
    print(f"  lr=0.02           compiles={counter.frame_count}  (recompile)")

    reset()
    tensor_counter = CompileCounter()

    def scale_tensor(x: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        return x * lr

    fn_t = torch.compile(scale_tensor, backend=tensor_counter)
    fn_t(x, torch.tensor(0.01))
    fn_t(x, torch.tensor(0.02))
    print(f"  tensor lr 0.01→0.02 compiles={tensor_counter.frame_count}")
    print("fix: keep mutating scalars as tensors inside compiled regions")


def main() -> None:
    print("torch.compile failure-mode lab")
    print(f"torch {torch.__version__}  (CPU, backend=eager / CompileCounter)")
    demo_graph_break_python_if()
    demo_graph_break_print()
    demo_recompile_shapes()
    demo_recompile_scalar()
    print("\nDone. For logs on a real model:")
    print('  TORCH_LOGS="graph_breaks,recompiles,guards" python train.py')


if __name__ == "__main__":
    main()
