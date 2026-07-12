"""MoE performance microbenchmarks on Modal for the Large MoE blog post.

Single-GPU benches that still expose the systems story:
  - analytical arithmetic intensity (NVLink vs IB balance points)
  - naive per-expert matmul vs ``torch._grouped_mm``
  - permute / gather traffic cost
  - aux-loss-free bias load-balancing simulation

Usage (from repo root)::

    uv run modal run playground/moe_perf_modal.py
    MOE_GPU=H100 uv run modal run playground/moe_perf_modal.py

Writes ``playground/moe_perf_results.json``. Figures via ``moe_perf_figures.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

GPU = os.environ.get("MOE_GPU", "A10G")

app = modal.App("moe-perf")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")


def analytical_ai() -> dict:
    """DeepSeek-V3-ish MoE layer: FLOPs and bytes per token for EP traffic."""
    hidden = 7168
    intermediate = 2048
    top_k = 8
    bytes_per_elem = 2  # BF16

    bytes_per_token = 2 * top_k * hidden * bytes_per_elem  # dispatch + combine
    # SwiGLU: gate/up/down ≈ 3 matmuls, 2 FLOPs per MAC
    flops_per_token = top_k * 3 * 2 * hidden * intermediate
    ai = flops_per_token / bytes_per_token

    hardware = {
        "nvlink_h100": {
            "bw_gbs": 900,
            "peak_tflops": 989,
            "balance_flop_per_b": 989e12 / 900e9,
        },
        "ib_400g_ndr": {
            "bw_gbs": 40,
            "peak_tflops": 989,
            "balance_flop_per_b": 989e12 / 40e9,
        },
    }
    return {
        "H": hidden,
        "I": intermediate,
        "top_k": top_k,
        "bytes_per_token": bytes_per_token,
        "flops_per_token": flops_per_token,
        "arithmetic_intensity": ai,
        "hardware": hardware,
        "nvlink_compute_bound": ai > hardware["nvlink_h100"]["balance_flop_per_b"],
        "ib_comm_bound": ai < hardware["ib_400g_ndr"]["balance_flop_per_b"],
    }


@app.function(gpu=GPU, image=image, timeout=900)
def bench() -> dict:
    import torch

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def median_ms(fn, warmup: int = 10, iters: int = 40) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        return times[len(times) // 2]

    results: dict = {
        "device": device_name,
        "gpu_slot": GPU,
        "analytical": analytical_ai(),
        "grouped_gemm": [],
        "permute": [],
        "load_balance": {},
    }

    # --- Grouped GEMM vs naive loop (torchtitan-style API) -------------------
    configs = [
        {"T": 4096, "H": 2048, "I": 1024, "E": 8, "label": "small_E8"},
        {"T": 4096, "H": 2048, "I": 1024, "E": 32, "label": "med_E32"},
        {"T": 8192, "H": 4096, "I": 2048, "E": 16, "label": "large_E16"},
    ]

    for cfg in configs:
        n_tokens, hidden, intermediate, n_experts = (
            cfg["T"],
            cfg["H"],
            cfg["I"],
            cfg["E"],
        )
        # Balanced token counts, aligned to 8 for BF16 Tensor Cores.
        base = (n_tokens // n_experts) // 8 * 8
        counts_list = [base] * n_experts
        leftover = ((n_tokens - base * n_experts) // 8) * 8
        for i in range(0, leftover, 8):
            counts_list[i // 8 % n_experts] += 8

        counts = torch.tensor(counts_list, dtype=torch.int32, device=device)
        # Length E: offs[i] = exclusive end of expert i (torchtitan convention).
        offs = torch.cumsum(counts, dim=0, dtype=torch.int32)
        total = int(offs[-1].item())
        # docs: offs[-1] must be < jagged dim length — pad one aligned row group.
        pad = 8
        x = torch.randn(total + pad, hidden, device=device, dtype=torch.bfloat16)
        # Weight layout [E, I, H] like torchtitan w1; matmul uses transpose → [E, H, I].
        w = torch.randn(n_experts, intermediate, hidden, device=device, dtype=torch.bfloat16)

        def naive(
            x=x,
            w=w,
            offs=offs,
            n_experts=n_experts,
            intermediate=intermediate,
        ):
            outs = []
            start = 0
            for e in range(n_experts):
                end = int(offs[e].item())
                if end > start:
                    outs.append(x[start:end] @ w[e].transpose(0, 1))
                start = end
            return torch.cat(outs, dim=0) if outs else x.new_empty(0, intermediate)

        def grouped(x=x, w=w, offs=offs):
            return torch._grouped_mm(x, w.transpose(-2, -1), offs=offs)

        with torch.no_grad():
            y_n = naive()
            y_g = grouped()
            # grouped_mm may leave rows beyond offs[-1] uninitialized — compare prefix.
            max_err = (y_n.float() - y_g[:total].float()).abs().max().item()

        ms_naive = median_ms(naive)
        ms_grouped = median_ms(grouped)
        flops = 2 * total * hidden * intermediate
        results["grouped_gemm"].append(
            {
                **cfg,
                "tokens_used": total,
                "ms_naive": ms_naive,
                "ms_grouped": ms_grouped,
                "speedup": ms_naive / ms_grouped if ms_grouped > 0 else None,
                "tflops_naive": (flops / (ms_naive * 1e-3)) / 1e12,
                "tflops_grouped": (flops / (ms_grouped * 1e-3)) / 1e12,
                "max_abs_err": max_err,
            }
        )

    # --- Permute / gather cost ----------------------------------------------
    for n_tokens, hidden, top_k in [(4096, 4096, 8), (8192, 7168, 8)]:
        x = torch.randn(n_tokens, hidden, device=device, dtype=torch.bfloat16)
        token_idx = torch.arange(n_tokens, device=device).repeat_interleave(top_k)
        perm = torch.randperm(n_tokens * top_k, device=device)
        sorted_token = token_idx[perm]
        scores = torch.rand(n_tokens * top_k, device=device, dtype=torch.bfloat16)

        def gather_scale(x=x, sorted_token=sorted_token, scores=scores):
            gathered = x.index_select(0, sorted_token)
            return gathered * scores[:, None]

        def multi_step(x=x, sorted_token=sorted_token, scores=scores):
            g1 = x.index_select(0, sorted_token)
            g2 = g1.to(torch.float32)
            g3 = g2 * scores[:, None].float()
            return g3.to(torch.bfloat16)

        ms_fused = median_ms(gather_scale)
        ms_multi = median_ms(multi_step)
        bytes_out = n_tokens * top_k * hidden * 2
        results["permute"].append(
            {
                "T": n_tokens,
                "H": hidden,
                "top_k": top_k,
                "intermediate_mb": bytes_out / 1e6,
                "ms_gather_scale": ms_fused,
                "ms_multi_step": ms_multi,
                "speedup": ms_multi / ms_fused if ms_fused > 0 else None,
                "gb_s_gather_scale": (bytes_out / (ms_fused * 1e-3)) / 1e9,
            }
        )

    # --- Aux-loss-free bias load balancing ----------------------------------
    torch.manual_seed(0)
    n_experts, steps, tokens_per_step, top_k = 64, 200, 4096, 8
    update_rate = 1e-3
    hist_no_bias: list[torch.Tensor] = []
    hist_bias: list[torch.Tensor] = []

    def topk_counts(s: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
        key = s if b is None else s + b
        idx = key.topk(top_k, dim=-1).indices
        return torch.bincount(idx.reshape(-1), minlength=n_experts).float()

    for _ in range(steps):
        logits = torch.randn(tokens_per_step, n_experts)
        logits[:, :4] += 2.5
        scores = torch.sigmoid(logits)
        hist_no_bias.append(topk_counts(scores))

    bias = torch.zeros(n_experts)
    for _ in range(steps):
        logits = torch.randn(tokens_per_step, n_experts)
        logits[:, :4] += 2.5
        scores = torch.sigmoid(logits)
        counts = topk_counts(scores, bias)
        hist_bias.append(counts.clone())
        delta = update_rate * torch.sign(counts.mean() - counts)
        bias += delta - delta.mean()

    def imbalance(hists: list[torch.Tensor]) -> dict:
        stack = torch.stack(hists[-50:])
        mean = stack.mean(dim=0)
        return {
            "cv": (mean.std() / mean.mean()).item(),
            "max_over_mean": (mean.max() / mean.mean()).item(),
            "dead_experts": int((mean < 1.0).sum().item()),
            "mean_tokens": mean.tolist(),
        }

    results["load_balance"] = {
        "E": n_experts,
        "top_k": top_k,
        "steps": steps,
        "update_rate": update_rate,
        "no_bias": imbalance(hist_no_bias),
        "aux_loss_free_bias": imbalance(hist_bias),
    }

    return results


@app.local_entrypoint()
def main() -> None:
    out = bench.remote()
    path = Path("playground/moe_perf_results.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"device: {out['device']}")
    print(f"AI: {out['analytical']['arithmetic_intensity']:.0f} FLOP/B")
    for row in out["grouped_gemm"]:
        print(
            f"grouped_gemm[{row['label']}]: "
            f"naive {row['ms_naive']:.2f}ms → grouped {row['ms_grouped']:.2f}ms "
            f"({row['speedup']:.2f}×)  err={row['max_abs_err']:.2e}"
        )
    for row in out["permute"]:
        print(
            f"permute T={row['T']} H={row['H']}: "
            f"multi {row['ms_multi_step']:.2f}ms → gather+scale {row['ms_gather_scale']:.2f}ms "
            f"({row['speedup']:.2f}×)  intermediate={row['intermediate_mb']:.0f}MB"
        )
    lb = out["load_balance"]
    print(
        f"load_balance CV: no_bias={lb['no_bias']['cv']:.3f} → "
        f"bias={lb['aux_loss_free_bias']['cv']:.3f}"
    )
    print(f"wrote {path}")
