---
title: "Roofline: The First Step of Any Performance Optimization"
date: 2026-07-11
tags: ["Performance", "Roofline", "GPU", "CUDA", "MFU", "Modal"]
categories: ["Engineering", "Performance"]
draft: false
---

# Roofline: The First Step of Any Performance Optimization

When you see MFU sitting at 20%, what's your first reaction?

Most people immediately reach for a profiler — hunt for the "slow kernel", look for bubbles in the timeline. But all of those reactions skip the single most important step in performance work: **figuring out which wall you actually hit**.

## 1. The Step Everyone Skips

In deep learning performance work, the most expensive mistake isn't "failing to find the optimal solution". It's picking the wrong direction from the start.

A real scenario:

> A dense model trains on A100s at 45% MFU. The team upgrades to H100s expecting a ~3× speedup. They get 1.5×, and MFU *drops* to 32%. Someone suspects the interconnect. Someone starts rewriting kernels. Someone blames the CUDA version. The real story: **the hardware changed, so the nature of the bottleneck changed.** Operators that were compute-bound on A100 got pushed into the bandwidth-bound regime on H100.

This is not a one-off. New hardware keeps changing the rules of the game while most of us keep reasoning with the old ones.

Roofline is the model that lets you see, at a glance, *which wall you're up against*. One 2D chart plus two ratios (MFU / MBU) answer two questions:

1. Is this operator compute-bound or bandwidth-bound?
2. How much headroom is left?

This post explains the model, grounds it in numbers for A100 / H100 / B200, backs it with experiments you can reproduce on [Modal](https://modal.com) for a few cents, and ends with a decision procedure you can copy verbatim.

## 2. Every Piece of Hardware Has Two Ceilings

Roofline comes from Williams, Waterman, and Patterson's classic 2009 CACM paper. It was designed for multicore CPUs and later adopted wholesale by NVIDIA, LBNL, and Intel for GPUs and HPC.

Its core insight fits in one sentence:

> Every piece of hardware has exactly two ceilings — compute (FLOP/s) and memory bandwidth (B/s). How fast a piece of code runs depends on which ceiling it hits first.

A traditional profiler tells you "this kernel took X milliseconds". It does *not* tell you: during those X milliseconds, was the GPU computing, or waiting for data?

Roofline is the minimal model that separates those two things.

## 3. Three Concepts

### Arithmetic intensity `I` — a property of the algorithm

```text
I = FLOPs / Bytes moved to and from memory        [FLOP/B]
```

The key point: **`I` is an intrinsic property of the algorithm/operator, independent of hardware.** Fix the algorithm and the precision, and `I` is fixed. Move the same GEMM to an H100 and `I` doesn't change — the *walls around it* move.

Classic examples (FP32):

| Operation | FLOPs | Bytes | I (FLOP/B) |
|---|---|---|---|
| SAXPY: `y = a*x + y` | 2N | 12N | 0.17 |
| Dot product: `sum(x*y)` | 2N | 8N | 0.25 |
| GEMM: `C = A·B`, square N×N | 2N³ | 12N² | N/6 |

The interesting bit: GEMM's intensity grows *linearly* with the matrix edge. That's why large matmuls are naturally compute-bound while element-wise ops are forever memory-bound — not because "the optimization wasn't done right", but because the algorithm dictates it.

NVIDIA's docs call this the *ops-to-byte ratio*; same thing.

### Performance ceiling `P` — the roof drawn by the hardware

```text
P(I) = min(I × PeakBandwidth, PeakFLOPS)
```

Two line segments:

- **Left segment (the slope):** bandwidth is the bottleneck; performance rises linearly with `I`.
- **Right segment (the plateau):** compute is the bottleneck; the ceiling is peak FLOP/s.

Where they meet is the **ridge point**:

```text
I_peak = PeakFLOPS / PeakBandwidth
```

Comparing an operator's `I` with `I_peak` classifies the bottleneck:

| Test | Bottleneck | Optimization direction |
|---|---|---|
| `I < I_peak` | memory-bound (slope) | raise `I`: fusion, reuse, fewer memory trips |
| `I > I_peak` | compute-bound (plateau) | raise throughput: TensorCores, lower precision, parallelism |
| `I ≈ I_peak` | balanced | do both, ranked by ROI |

### One chart, one glance

![The Roofline model: two ceilings, one ridge point](roofline_concept.svg)

How to read it:

1. Take the operator's `I`, place it on the x-axis.
2. Go straight up until you hit the roof.
3. That height is the operator's *theoretical* ceiling on this hardware.
4. Measured performance ÷ ceiling = utilization.

That's the whole model.

## 4. Modern GPUs: Where Is the Roof?

Numbers from NVIDIA datasheets (theoretical peaks, dense — no sparsity):

| GPU | Peak FLOPS (BF16) | HBM bandwidth | I_peak (BF16) |
|---|---|---|---|
| V100 SXM2 | 125 TFLOPS | 900 GB/s | ~139 |
| A100 SXM 80GB | 312 TFLOPS | 2039 GB/s | ~153 |
| H100 SXM | 989 TFLOPS | 3.35 TB/s | ~295 |
| H200 SXM | 989 TFLOPS | 4.8 TB/s | ~206 |
| B200 (Blackwell) | 2250 TFLOPS | 8 TB/s | ~281 |
| GB200 NVL72 | 2500 TFLOPS/GPU | 8 TB/s | ~312 |

This one table captures the central tension of five years of GPU evolution — easier to see as a picture:

![Compute grows faster than bandwidth, so the ridge point drifts right](gpu_evolution.svg)

> **Compute is growing far faster than bandwidth.** Since V100, peak BF16 compute has grown 18× while bandwidth grew 8.9×. The ridge point moved from 139 (V100) to 295 (H100) to nearly 300 (B200). The "memory-bound" territory keeps expanding; more and more operators get pushed onto the slope.

That single trend explains why FlashAttention, `torch.compile`, and FP8 GEMMs have been promoted so hard in recent years — they're all doing the same thing: **pushing an operator's `I` back to the right of the ridge point.**

One counter-intuitive fact about FP8: H100's FP8 peak is 1979 TFLOPS, so `I_peak(FP8) ≈ 591`. Halving the precision doubles compute but leaves bandwidth unchanged — nearly every element-wise op gets shoved to the far left of the slope. That's why FP8 only pays off for large GEMMs.

## 5. MFU: The Metric for Training

MFU (Model FLOPs Utilization) comes from Google's PaLM paper:

```text
MFU = (model FLOPs per second, achieved) / (peak FLOPs per second, hardware)
```

Note the denominator is a *rate* — FLOP/s over FLOP/s, not "total FLOPs per step".

**How do you estimate FLOPs per step?**

For a Transformer LLM, the standard approximation is `6·P·T` (Kaplan 2020 / Chinchilla):

- `P`: trainable parameters
- `T`: tokens per step (batch × sequence length)
- The factor 6: forward costs `2PT`, backward costs `4PT` (one `2PT` each for parameter grads and activation grads)

Long sequences need an attention correction term:

```text
FLOPs/step ≈ 6·P·T + 12·L·H·Q·T·S      (L layers, H heads, Q head dim, S seq len)
```

Below seq 1024 the second term is negligible; above 4096 it can be 30–50% of the total, and plain `6PT` will underestimate.

**Reference MFU levels from public training reports:**

| Run | Hardware | MFU |
|---|---|---|
| PaLM 540B | TPU v4 | 46.2% |
| GPT-3 175B | V100 | ~21.3% |
| Megatron-Turing NLG 530B | A100 | 30.2% |
| Llama 2 70B | A100 | ~40–45% |
| DeepSeek-V3 671B (MoE) | H800 | ~40% |

Mapped to your own runs:

| MFU | Status | Notes |
|---|---|---|
| < 20% | something is clearly wrong | communication, CPU, or dataloader bottleneck |
| 20–35% | mediocre | obvious headroom |
| 35–50% | good | mainstream range for large LLM training |
| > 50% | excellent | only carefully tuned dense training gets here |
| > 60% | rare | usually extreme packing + custom kernels |

**MFU vs HFU.** The PaLM paper also defines HFU (Hardware FLOPs Utilization), which counts gradient-checkpointing recompute FLOPs in the numerator:

- MFU: only the FLOPs the model *should* do — measures **algorithmic efficiency**
- HFU: includes recompute — measures **whether the hardware is busy**

HFU typically runs ~33% higher than MFU. MFU is the one that matters. When someone reports "60% MFU", the first question to ask is: *does that include recompute?*

## 6. MBU: The Metric for Inference

MBU (Model Bandwidth Utilization) was popularized by Databricks / MosaicML:

```text
MBU = (achieved memory bandwidth) / (peak memory bandwidth)
```

Why does inference care about MBU more than MFU? Because in the decode phase at batch=1, almost all the time goes to reading parameters out of HBM — MBU directly determines tokens/s.

**A concrete calculation.** Llama-70B (`P = 7×10¹⁰`) doing batch=1 BF16 decode on an H100 (3.35 TB/s):

```text
bytes per token ≈ 2 × 7e10        = 140 GB
time per token  ≥ 140 GB / 3.35 TB/s ≈ 41.8 ms
ceiling         ≈ 24 tokens/s
```

That's the ceiling for *all* inference optimization on that setup. Without speculative decoding, continuous batching, or quantization, no kernel wizardry breaks 24 tok/s — because just *moving the weights from HBM to the compute units* takes 41.8 ms.

**Reference MBU levels:**

| MBU | Status |
|---|---|
| < 40% | clear kernel-launch / CPU bottleneck |
| 40–60% | mediocre |
| 60–80% | good — the vLLM / TensorRT-LLM target range |
| > 80% | excellent |

## 7. Arithmetic Intensity of Common DL Operators

Horace He's [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) gives the sharpest framing: every DL program's time is divided among three costs — **compute, memory, overhead**. Roofline covers the first two; overhead (kernel launches, Python, graph capture) must be handled separately.

This table is worth keeping around:

| Operator | I (BF16, FLOP/B) | Verdict on H100 | Optimization |
|---|---|---|---|
| Element-wise (Add, GELU) | 0.5–1.5 | Memory-bound | fusion |
| LayerNorm / RMSNorm | ~1 | Memory-bound | fused kernels (Apex/Triton) |
| Softmax | ~0.75 | Memory-bound | online softmax / FA fusion |
| Attention (vanilla) | ~64 | Memory-bound | FlashAttention |
| Attention (FlashAttention) | ~200 | near balanced | already optimized |
| GEMM (M=N=K=4096) | ~1365 | Compute-bound | TensorCore alignment, FP8 |
| GEMM (decode, M=1, K=N=4096) | ~2 | Memory-bound | quantization, continuous batching |
| Embedding lookup | ~0 | Memory-bound | low-precision storage, caching |

Look at the two GEMM rows. Both are "GEMM", but the shapes differ, so one is strongly compute-bound and the other strongly memory-bound. **This is the root cause of "training is compute-bound, decode is memory-bound"** — not a difference in operator kind, but a difference in shape that changes `I` entirely.

**The right way to think about FlashAttention.** Vanilla attention writes the S×S score matrix to HBM and reads it back for softmax — HBM traffic is O(S²). FlashAttention keeps that matrix in SRAM — HBM traffic drops to O(S). The FLOPs don't change at all; the *bytes* do. That's why it saves memory and time simultaneously: "reducing memory traffic" and "raising `I`" are the same statement.

## 8. Measuring It Yourself: Experiments on Modal

Theory tables are nice, but the whole point of Roofline is that it's *checkable with a stopwatch*. I wrote a small benchmark that runs every operator from the table above on a cloud GPU via [Modal](https://modal.com), counts FLOPs and bytes analytically, and drops each op onto the roofline. Full code: [`playground/roofline_modal.py`](https://github.com/duoan/duoan.github.io/blob/main/playground/roofline_modal.py) and [`playground/roofline_figures.py`](https://github.com/duoan/duoan.github.io/blob/main/playground/roofline_figures.py).

The skeleton is minimal — Modal makes "run this function on a GPU" a decorator:

```python
import modal

app = modal.App("roofline")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")

@app.function(gpu="A10G", image=image, timeout=1200)
def bench() -> dict:
    import torch

    def time_op(fn, warmup=10, iters=50):
        """Median wall time of fn() in seconds, via CUDA events."""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3)
        return sorted(times)[len(times) // 2]
    ...
```

For each op I record measured seconds plus *analytic* FLOPs and bytes, e.g.:

```python
# Elementwise add: 1 FLOP/elem, read 2 elems, write 1
s = time_op(lambda: torch.add(x, y))
record("Elementwise add", s, flops=n, bytes_moved=3 * n * esize)

# Decode-shaped GEMV: M=1, K=N=8192 — same "GEMM", wildly different I
s = time_op(lambda: a @ b)   # a: (1, 8192), b: (8192, 8192)
record("decode GEMV", s, flops=2 * k * k, bytes_moved=(k * k + 2 * k) * esize)
```

Then:

```bash
uv run modal run playground/roofline_modal.py
```

My Modal account is on the free tier, which doesn't offer H100s, so the numbers below come from an **NVIDIA A10G** (datasheet: 70 TFLOPS BF16 dense, 600 GB/s GDDR6, so `I_peak ≈ 117`). The script takes `ROOFLINE_GPU=H100` if your account has one — the *shape* of the results is the same on any GPU; only the ridge point moves.

### The measured roofline

![Measured roofline on an NVIDIA A10G](roofline_measured.svg)

Everything the theory said shows up in the data (BF16 throughout):

| Op | I (FLOP/B) | Achieved | Utilization |
|---|---|---|---|
| memcpy 1 GiB (bandwidth ceiling) | — | 472 GB/s | 79% of 600 GB/s |
| GEMM 8192³ (compute ceiling) | 2731 | 65.9 TFLOP/s | 94% of 70 TFLOPS |
| Elementwise add | 0.17 | 477 GB/s | **79% MBU**, 0.08 TFLOP/s |
| GELU | 2.5 | 477 GB/s | 79% MBU |
| RMSNorm 8192×8192 | 1.0 | 315 GB/s | 53% MBU |
| Softmax 8192×8192 | 1.25 | 476 GB/s | 79% MBU |
| GEMM M=1, K=N=8192 (decode) | ~1.0 | 438 GB/s | **73% MBU**, 0.44 TFLOP/s |
| Attention vanilla, S=4096 | 31.5 | 3.2 TFLOP/s | 4.5% MFU |
| FlashAttention, S=4096 | 2048 | 61.4 TFLOP/s | 88% MFU |

Three things worth staring at:

1. **The memory-bound ops sit exactly on the slope.** Add, GELU, softmax, and the decode GEMV all achieve ~440–477 GB/s — pinned to the measured bandwidth roof — while their FLOP/s are pitiful. They are *done*. No kernel tuning will make `torch.add` faster; only fusion (fewer trips to memory) changes anything.
2. **The decode GEMV is the punchline of the whole post.** Same `matmul` call as the 8192³ GEMM, but at M=1 it lands at `I ≈ 1` and runs at 0.44 TFLOP/s — a **150× gap** from the square GEMM — while achieving 73% MBU. The kernel is not slow. The *shape* is bandwidth-starved.
3. **FlashAttention vs vanilla is a 19× wall-clock gap** on this shape (S=4096: 86.6 ms → 4.5 ms) with identical FLOPs. Vanilla attention drags the S² score matrix through HBM about four times; FlashAttention keeps it in SRAM, moving its intensity from ~32 to ~2048 — from the slope to the plateau, at 88% MFU.

### Same op, different shape

![GEMM throughput vs shape](gemm_shape_sweep.svg)

The GEMM sweep makes the "intensity is a property of the shape" point directly: at 256³ the GPU manages 0.9 TFLOP/s — barely 1% of peak — and only approaches the roof from 2048³ onward. Datasheet peaks require shapes the kernel likes.

### Does a *model-level* roofline make sense?

A fair objection at this point: everything above is per-operator. Can you put a whole *model* on the roofline?

In general, no — and it's worth understanding why. A model is a mixture of operators, and a whole-model "average intensity" (total FLOPs ÷ total bytes) is dominated by the big GEMMs, while wall-clock time can be dominated by memory-bound ops. Averaging hides exactly the distinction Roofline exists to make. For a heterogeneous op mix, the tools are the per-op table above and a profiler.

But there is one case where model-level roofline is not just valid but *the sharpest tool available*: **LLM decode**. Every decode step reads every weight exactly once (2 bytes/param in BF16) and does ~2 FLOPs per parameter *per sequence in the batch*. So for the whole model:

```text
I_model ≈ (2 · P · B) / (2 · P) = B      — arithmetic intensity IS the batch size
```

The x-axis of the roofline becomes the batch size, and the model predicts something very concrete: throughput should grow *linearly and almost for free* with batch until `B ≈ I_peak`, then flatten into the compute roof. That predicted knee is the **critical batch size**, and it's a number a profiler cannot hand you — a profiler describes the run you did, not the run you should be doing.

I tested this with a 1.4B-parameter MLP-only model (16 blocks of 4096 → 11008 → 4096, BF16) on Modal, sweeping the decode batch from 1 to 512 (`modal run playground/roofline_modal.py::decode`). This run landed on an NVIDIA A10 (125 TFLOPS BF16 / 600 GB/s ⇒ critical batch ≈ 208):

![Model-level roofline: decode throughput vs batch size](decode_batch_sweep.svg)

The measured curve does exactly what the napkin math says:

- **Batch 1:** 166 tok/s against a bandwidth ceiling of `600 GB/s ÷ 2.9 GB of weights ≈ 208 tok/s` — 80% MBU before writing a single line of custom code.
- **Batch 1 → 128:** throughput grows 90× while per-step latency only rises from 6.0 ms to 8.5 ms. Batching is nearly free — the weights are being read anyway; extra sequences ride along.
- **Batch 128 → 512:** the curve bends onto the plateau. 4× more batch buys only 1.5× more throughput, and latency now grows 2.7×. The free lunch is over; you've crossed the ridge.

**What about training?** Then "model-level" means forward *plus* backward, and the same accounting still works — it's just the MFU calculation from section 5 in disguise. FLOPs per step become `6·P·T` (forward `2PT`, backward `4PT`), while weight traffic stays `O(P)`: read the weights in forward, read them again in backward, write gradients, touch optimizer state. So:

```text
decode:    I ≈ 2PB / 2P  = B          — intensity is the batch size
training:  I ≈ 6PT / kP  ∝ T          — intensity is tokens per step
```

`T` in real training is 10⁵–10⁷ tokens per step, which puts training's model-level intensity far to the right of any ridge point. That's the napkin derivation of "training is naturally compute-bound, decode is naturally memory-bound" — and of why the right lens is MFU for training but MBU for decode.

Same model, same GPU, but now each step is `forward + backward` (`modal run playground/roofline_modal.py::train`), sweeping tokens/step `T` from 1 to 4096:

![Model-level roofline: training throughput vs tokens per step](train_tokens_sweep.svg)

The shape mirrors decode, but the accounting changes because backward triples weight traffic (read in fwd, read again in bwd, write grads):

- **T = 1:** 53 tok/s and 0.46 TFLOP/s — roughly 3× slower than the decode run at batch = 1 (166 tok/s), because the same weights traverse HBM three times instead of once. Still on the slope; MBU-dominated.
- **T = 1 → 128:** throughput climbs 97× while step time only goes from 19 ms to 25 ms. Extra tokens are nearly free — exactly the `I ∝ T` prediction.
- **T = 1024 → 4096:** the curve bends onto the compute plateau at ~8,000 tok/s and ~70 TFLOP/s — **56% MFU** on this A10 (125 TFLOPS datasheet peak). Real LLM training runs at `T` two to four orders of magnitude larger, so it lives on this plateau; the question is how close to the roof you are, not which wall you hit.

Note the same caveat applies in both directions: this holds at the *aggregate* level, while individual memory-bound ops (norms, softmax, activations) inside the step still live on the slope — which is exactly what the per-op section was for.

So the answer to "model-level or profiler?" is: **they answer different questions.** The model-level roofline is a *forward-looking* calculation — it tells you the ceiling, the critical batch, and whether an optimization *can possibly* help, before you run anything. The profiler is *backward-looking* — it tells you where the time actually went in the run you made, which is what you need when the model is a heterogeneous mix and the aggregate numbers stop being trustworthy. Napkin roofline first, per-op roofline second, profiler third.

## 9. The Decision Procedure

Given a piece of code, the standard flow:

```text
Step 1  Measure: get achieved performance
  ├─ time: nsys / torch.profiler / triton.testing.do_bench
  └─ torch.cuda.synchronize() around the timed region

Step 2  Compute: derive I
  ├─ count FLOPs: 6PT formula or fvcore
  └─ count Bytes: 2P-per-token formula or nsys memory profile

Step 3  Look up: hardware I_peak (matching precision!)

Step 4  Classify: I vs I_peak
  ├─ I < I_peak / 2  → strongly memory-bound
  ├─ I ≈ I_peak      → balanced
  └─ I > I_peak × 2  → strongly compute-bound

Step 5  Verify: compute MFU / MBU
  ├─ MFU high → already compute-bound → precision / hardware-level work
  ├─ MBU high → already memory-bound → fusion / reuse
  └─ both low → CPU / launch / comms → torch.compile + CUDA Graphs
```

Three canonical profile shapes:

- **Shape A: high MFU, low MBU — compute-bound.** Directions: lower precision (BF16 → FP8), TensorCore alignment, sparsity, MoE. Typical: GEMM-heavy large-batch training.
- **Shape B: high MBU, low MFU — memory-bound.** Directions: operator fusion, FlashAttention, KV-cache reuse, quantized storage. Typical: LLM decode.
- **Shape C: both low — overhead-bound.** Directions: `torch.compile`, CUDA Graphs, fewer Python round-trips, bigger batches. Typical: an nsys timeline that's mostly blank — GPU idle → burst → idle → burst.

Horace He's description of shape C is hard to beat: *"You have a GPU doing nothing, then suddenly a burst of activity, then nothing again."*

## 10. Five Traps

**Trap 1: "Higher MFU is always better."** No. 40% MFU with stable training and robust hyperparameters can beat 60% MFU that requires padding every sequence to 4096 and burning half the compute. Watch *effective tokens/s*.

**Trap 2: "Roofline has exactly two segments."** The original model draws compute and DRAM only. Real hardware adds L2, shared memory, NVLink, InfiniBand — multiple roofs. LBNL's Hierarchical Roofline draws them all; distributed training needs that extension.

**Trap 3: "I doesn't depend on batch size."** True for element-wise ops, completely false for GEMM. Decode means batch=1 means M=1 means `I` collapses. This is exactly why batching is the silver-bullet inference optimization.

**Trap 4: "Peak FLOPS is the whitepaper number."** The whitepaper number is a theoretical peak *under conditions*: TensorCores enabled, shapes aligned (M/N/K multiples of 8/16), matching precision. And the sparse peaks (the ":2:4" numbers) require a sparsity pattern dense models don't have. A100's "624 TFLOPS BF16" is the sparse number — dense Roofline math uses 312. H100 likewise: 989, not 1979. The A10G sweep above makes the shape condition concrete: the same cuBLAS matmul spans 0.9 to 65.9 TFLOP/s depending only on its dimensions.

**Trap 5: "MFU is high, so I can ignore MBU."** High MFU only says the kernels are busy. Training is MFU-dominated, inference is MBU-dominated — you need both to see the full picture.

## 11. A Real Incident: Why Did H100 Make It Slower?

Back to the opening scenario. MFU went from 45% on A100 to 32% on H100; wall-clock improved only 1.5× against a 3.2× compute upgrade.

The Roofline diagnosis:

![A100 vs H100 rooflines: the same operator lands on different sides of the ridge](a100_vs_h100.svg)

- A100: `I_peak ≈ 153`
- H100: `I_peak ≈ 295` — the ridge point moved right by almost 2×
- Operators that sat right of the A100 ridge (compute-bound) landed *left* of the H100 ridge (memory-bound) — the chart shows an operator at `I = 200` sitting on the A100 plateau but on the H100 slope
- Bandwidth improved only 1.65× (2.0 → 3.35 TB/s) against 3.2× compute

The fix wasn't a bug hunt: FlashAttention, larger batches, FP8 GEMMs — pushing `I` back to the right of the new ridge.

That's the value of Roofline. It doesn't tell you "there's a bug here". It tells you **"the direction you were about to optimize in is wrong."**

## 12. Closing

Roofline isn't a silver bullet. It can't explain communication bottlenecks in detail, doesn't cover recompilation overhead from dynamic shapes, and says nothing about load balancing across nodes.

But it is the *starting point* of all performance work.

Five minutes of Roofline before opening the profiler can save days or weeks of optimizing in the wrong direction. Because the most expensive mistake in performance engineering isn't failing to find a solution — it's acting before locating the problem.

## References

**Roofline, original and extensions**

- Williams et al., [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785), CACM 2009
- [LBNL Roofline Performance Model](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)
- NVIDIA, [Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)

**MFU and training performance**

- Chowdhery et al., [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311), 2022
- Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), 2020
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla), 2022

**MBU and inference performance**

- Databricks / MosaicML, [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices), 2023

**Operator-level analysis**

- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) / [FlashAttention-2](https://arxiv.org/abs/2307.08691)

If this post was useful, forward it to the colleague currently staring at 20% MFU.
