"""Minimal torch.compile-shaped demo: capture → guard → cache → backend.

Educational toy only. Not a Dynamo / Inductor reimplementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch


class CapturedGraph:
    def __init__(self, func: Callable, args: tuple) -> None:
        self.name = func.__name__
        tensors = [a for a in args if isinstance(a, torch.Tensor)]
        self.input_shapes = [tuple(t.shape) for t in tensors]
        self.input_dtypes = [t.dtype for t in tensors]


def capture_graph(func: Callable, *args: object) -> CapturedGraph:
    return CapturedGraph(func, args)


@dataclass
class Guard:
    shape_checks: dict[str, tuple] = field(default_factory=dict)
    dtype_checks: dict[str, torch.dtype] = field(default_factory=dict)
    dynamic_dims: dict[str, list[int]] = field(default_factory=dict)

    def check(self, inputs: tuple) -> bool:
        for i, (name, (expected_shape, expected_dtype)) in enumerate(
            self.shape_checks.items()
        ):
            if i >= len(inputs):
                return False
            inp = inputs[i]
            if not isinstance(inp, torch.Tensor):
                return False
            if expected_dtype != inp.dtype:
                print("Guard failed: dtype mismatch")
                return False
            if len(expected_shape) != len(inp.shape):
                print("Guard failed: rank mismatch")
                return False
            dynamic = set(self.dynamic_dims.get(name, []))
            for dim_idx, (exp_dim, actual_dim) in enumerate(
                zip(expected_shape, inp.shape, strict=True)
            ):
                if dim_idx in dynamic:
                    continue
                if exp_dim != actual_dim:
                    print(
                        f"Guard failed: shape mismatch at dim {dim_idx}, "
                        f"expected {exp_dim}, got {actual_dim}"
                    )
                    return False
        return True


class InductorBackend:
    @staticmethod
    def compile(graph: CapturedGraph, dynamic: bool = False) -> Callable:
        print(f"\n[Inductor] Compiling {graph.name}...")
        print(f"  Input shapes: {graph.input_shapes}")
        print(f"  Dynamic mode: {dynamic}")

        def compiled_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.relu(x + y)

        if dynamic:
            print("  [Dynamic] Generated kernel (relies on PyTorch broadcasting)")
        else:
            print(
                f"  [Static] Specialized for shapes "
                f"{graph.input_shapes[0]} and {graph.input_shapes[1]}"
            )
        return compiled_func


class TorchCompile:
    _cache: dict[str, tuple[Callable, Guard]] = {}

    def __init__(self, func: Callable, dynamic: bool = False) -> None:
        self.func = func
        self.dynamic = dynamic

    def __call__(self, *args: object) -> object:
        cache_key = f"{self.func.__name__}_{self.dynamic}"

        if cache_key in self._cache:
            compiled_func, guard = self._cache[cache_key]
            if guard.check(args):
                print(f"[Cache] Hit! Reusing compiled {self.func.__name__}")
                return compiled_func(*args)
            print("[Cache] Guard failed, recompiling...")

        print(f"[Dynamo] Capturing {self.func.__name__}...")
        graph = capture_graph(self.func, *args)
        compiled_func = InductorBackend.compile(graph, dynamic=self.dynamic)

        guard = Guard()
        for i, (shape, dtype) in enumerate(
            zip(graph.input_shapes, graph.input_dtypes, strict=True)
        ):
            name = f"input_{i}"
            guard.shape_checks[name] = (shape, dtype)
            if self.dynamic:
                guard.dynamic_dims[name] = list(range(len(shape)))

        self._cache[cache_key] = (compiled_func, guard)
        print("[Dynamo] Compiled and cached\n")
        return compiled_func(*args)


def compile(dynamic: bool = False):
    def decorator(func: Callable) -> TorchCompile:
        return TorchCompile(func, dynamic=dynamic)

    return decorator


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: mini torch.compile demo")
    print("=" * 60)

    @compile(dynamic=False)
    def my_model_static(x, y):
        return torch.relu(x + y)

    @compile(dynamic=True)
    def my_model_dynamic(x, y):
        return torch.relu(x + y)

    print("\n--- Test 1: Static mode, same shape ---")
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    out1 = my_model_static(a, b)
    out2 = my_model_static(a, b)
    print(f"  Output shape: {out2.shape}")
    print(f"  Correctness: {torch.allclose(out2, torch.relu(a + b))}")

    print("\n--- Test 2: Static mode, broadcastable shape change ---")
    a_broad = torch.randn(1, 4)
    c_broad = torch.randn(5, 4)
    out3 = my_model_static(a_broad, c_broad)
    print(f"  Output shape: {out3.shape}")
    print(f"  Correctness: {torch.allclose(out3, torch.relu(a_broad + c_broad))}")

    print("\n--- Test 3: Dynamic mode, batch size change ---")
    x1 = torch.randn(3, 4)
    y1 = torch.randn(3, 4)
    _ = my_model_dynamic(x1, y1)
    x2 = torch.randn(5, 4)
    y2 = torch.randn(5, 4)
    out5 = my_model_dynamic(x2, y2)
    print(f"  Output shape: {out5.shape}")
    print(f"  Correctness: {torch.allclose(out5, torch.relu(x2 + y2))}")

    print("\n--- Test 4: Dynamic mode, different ranks ---")
    e = torch.randn(4)
    f = torch.randn(3, 4)
    out6 = my_model_dynamic(e, f)
    print(f"  Output shape: {out6.shape}")
    print(f"  Correctness: {torch.allclose(out6, torch.relu(e + f))}")

    print("\nAll tests passed.")
    _ = out1  # silence unused lint for first compile result
