---
title: "Distributed Training Failure Runbook: Reproducing the Failures That Actually Bite"
date: 2026-07-13
tags: ["PyTorch", "DDP", "Distributed Training", "NCCL", "Debugging", "Modal", "Transformer"]
categories: ["Engineering", "Performance"]
draft: false
cover:
  image: failure_taxonomy.svg
  alt: "Taxonomy of distributed training failures: numerics, resources, and systems"
  relative: true
---

# Distributed Training Failure Runbook: Reproducing the Failures That Actually Bite

A war-room runbook for NaNs, loss spikes, silent rank desync, memory growth, stragglers, host-skewed nodes, empty-batch NCCL hangs, and throughput cliffs — reproduced on a real (tiny) Transformer LM under DDP/NCCL. No `sleep()`, no `tensor * 1e3`, no `param.add_` cartoons.

## TL;DR

Distributed jobs rarely fail cleanly. They **look almost healthy** while burning GPU-hours: loss spikes that recover, ranks that quietly disagree on weights, one long-doc bucket pacing everyone, or a rank that `continue`s past an empty pack and freezes NCCL.

This lab uses one shared model — **`TinyTransformerLM`** (causal attention + MLP + LayerNorm + tied embeddings) — and triggers failures from **code patterns that survive code review**:

| Failure | Real bug pattern | Modal 2×A10G / NCCL signal |
|---|---|---|
| NaN (7 recipes) | missing attn scale under fp16, fully-padded softmax, AMP w/o GradScaler, hand-rolled masked NLL, empty valid-token mean, Adam poison, DDP contagion | **7/7** triggered |
| Loss spike | z-loss / aux coefficient `100` instead of `1e-4` | **~180–190×** vs median |
| Silent drift | rank0-only `clip_grad` after allreduce; rank0-only EMA→student copy-back | params diverge; loss stays finite |
| Memory leak | unbounded on-device logits list “for later top-k logging” | CUDA alloc grows; ring-buffer control flat |
| Straggler | one rank stuck on long-doc bucket (`T=256` vs `T=16`) | token imbalance **16×**; measure fwd time too |
| Bad node (host skew) | persistent CPU tokenize/preprocess inflation on one rank | local host timer flagged |
| Collective hang | `if valid_tokens==0: continue` skips DDP backward | peer blocks in NCCL/DDP |
| Throughput cliff | tokens/rank sweep; real grad allreduce | tiny packs ≪ peak tokens/s |

```bash
# Modal (default: 2×A10G, NCCL):
uv run modal run playground/dist_failure_modal.py

# Local CPU/gloo:
uv run python playground/dist_failure_modal.py

uv run python playground/dist_failure_figures.py \
  --results playground/dist_failure_results.json \
  --out content/posts/distributed-training-failure-runbook
```

Code: [`playground/dist_failure_modal.py`](https://github.com/duoan/duoan.github.io/blob/main/playground/dist_failure_modal.py) · Results: [dist_failure_results.json](./dist_failure_results.json)

![Failure taxonomy](./failure_taxonomy.svg)

## Design rule: failures must come from the model path

If the only way to demo a failure is `time.sleep`, `x * 1e3`, or `p.add_(0.01)`, you have not captured the failure — you have drawn a cartoon of it.

Every case here is a **pattern you will find in production training code**:

- wrong attention / mask math in a custom kernel path
- aux-loss config typos (z-loss weight off by 10⁶)
- `if is_main:` around clip / EMA / step
- debug logging that retains GPU tensors
- length-bucket imbalance
- empty microbatch control-flow around DDP
- tokens/rank too small for the grad bucket

The model is small so the lab fits in a Modal A10G pair. The **bugs are not**.

Related reading: [DDP performance](/posts/learning-ddp-performance-tuning-on-one-gpu/), [variable-length skew](/posts/why-variable-sequence-length-breaks-ddp-throughput/), [ARGUS fail-slow](/posts/argus-tracing-at-10000-gpu-scale/), [profiling E2E](/posts/profiling-pytorch-training-end-to-end/).

### Lab setup

- **Modal `A10G:2`**, PyTorch `2.11+cu130`, **NCCL**, `world_size=2`
- Shared model: `TinyTransformerLM` (2–4 layers, dim 64–256, causal LM loss)
- Local fallback: CPU + `gloo` via `uv run python …` (fp16 overflow recipes need CUDA)

## Triage first

![Triage flowchart](./runbook_flowchart.svg)

1. **Hard fail or soft degrade?** NaN / NCCL hang / OOM vs spikes / drift / slow steps.
2. **Per-rank or global?** Collect rank-local loss, forward ms, host ms, alloc, checksums **before** blaming the fabric.

Always-on heartbeat (every rank):

```python
print(
    f"rank={rank} step={step} loss={loss.item():.4g} "
    f"finite={torch.isfinite(loss).item()} "
    f"fwd_ms={fwd_ms:.1f} host_ms={host_ms:.1f} alloc_mb={alloc_mb:.1f}",
    flush=True,
)
```

---

## 1. NaNs — a catalog on the LM

NaN is not one bug. The lab runs **seven recipes** against `TinyTransformerLM`:

| Recipe | Production pattern | Where it breaks |
|---|---|---|
| `attn_missing_scale` | custom attn forgot `1/sqrt(d_h)` (+ bad QKV scale) under fp16 | softmax / logits |
| `fully_padded_softmax` | packed row that is 100% padding | softmax(-inf,…)=NaN |
| `amp_no_scaler` | `autocast(fp16)` without `GradScaler` | loss / grads / params |
| `handrolled_masked_nll` | `mask * log_softmax` on a fully-padded row | `0*(-inf)` |
| `empty_valid_token_mean` | LM token-mean when valid count is 0 | ±Inf / NaN |
| `adam_moment_poison` | AMP overflow stepped into AdamW | optimizer state forever |
| `ddp_nan_contagion` | one rank’s all-pad pack → allreduce | every rank |

![NaN catalog](./nan_catalog.svg)

**Lab:** **7/7** triggered on Modal. Healthy AdamW+clip train loop stays finite. (CPU/gloo will miss the fp16 overflow recipes — that is expected.)

### Debug

1. `isfinite(loss)` **and** grads **before** `optimizer.step()` (Adam remembers a single NaN).
2. `all_reduce(has_nan)` to find the first offender before contagion.
3. Bisect: data/mask → attention → AMP/scaler → optimizer state.
4. After an incident, **reset Adam moments** or restore a clean checkpoint.

### Fix

- Prefer SDPA / flash-attn; never ship unscaled custom attn in fp16.
- Drop empty rows before forward; use `cross_entropy(..., ignore_index=-100)`.
- GradScaler (or BF16); skip nonfinite steps.

---

## 2. Loss spikes — z-loss / aux weight typo

**Pattern:** CE is fine; a secondary term (z-loss, router aux, load-balance) is configured as `100` instead of `1e-4` on some runs/steps. The dashboard shows a vertical spike; CE alone would not.

```python
ce = F.cross_entropy(...)
z_loss = logits.float().pow(2).mean()   # PaLM-style z-loss
loss = ce + z_weight * z_loss           # z_weight typo: 100 vs 1e-4
```

![NaN strip + loss spikes](./nan_and_loss_spike.svg)

**Lab:** spikes at steps `{20, 28}` reach **~180–190×** the healthy median. CE-only stays calm — if you only log the blended scalar, you will chase data ghosts.

### Debug

1. Log **CE and aux separately** — never one blended scalar only.
2. Diff spike steps against config / feature flags / resume.
3. Clip does not save you from a 100× aux term in the reported loss.

### Fix

- Unit-test loss scales in CI (`aux_weight < 1e-2`).
- Quarantine configs that change aux weights without a canary.

---

## 3. Silent numerical drift — rank-gated updates

Toy `param.add_` is not the bug. These are:

| Recipe | Pattern | Diverges |
|---|---|---|
| `rank0_only_grad_clip` | `if rank==0: clip_grad_norm_(...)` **after** DDP allreduce | **student parameters** |
| `rank0_only_ema_copyback` | mean-teacher / KD: `if rank==0: student.load_state_dict(ema)` | **student parameters** |

![Drift and memory leak](./drift_and_memory_leak.svg)

After `backward()`, grads are identical across ranks. Clipping only on rank 0 means rank 0 steps with a different grad than everyone else. EMA copy-back only on rank 0 is the same class of mistake in distillation stacks.

**Lab:** both recipes trigger; all-rank clip control stays synced; loss remains finite (dashboards stay green).

### Debug / fix

- Checksum **parameters** across ranks every K steps.
- Grep `is_main` / `rank == 0` around clip, unscale, `step`, `load_state_dict`, EMA.
- Broadcast after any intentional rank0-only write — or don’t do it.

---

## 4. Memory leaks — debug logits retention

**Pattern:** while prototyping top-k / sampling dumps:

```python
debug_logits.append(logits.detach())  # unbounded, still on GPU
```

**Lab:** leaky path grows CUDA/retained MB; fixed path keeps a **length-1 ring** and stays flat.

### Debug / fix

- Plot `torch.cuda.memory_allocated` every N steps.
- Bounded **CPU** ring buffers; log scalars, not full logits.
- `empty_cache` is not a fix for a retain.

---

## 5. Stragglers — long-doc vs short-doc bucket

**Pattern:** rank 1 draws `T=256`, peers stay on `T=16` after onset. Attention is `O(T²)` — this is the same family as [variable-length DDP skew](/posts/why-variable-sequence-length-breaks-ddp-throughput/), not a `sleep`.

Measure **forward time before backward** (local compute) **and** tokens/step. Step wall time alone lies because DDP makes everyone wait; on fast GPUs the token imbalance (here **16×**) is often the clearer signal before fwd-time ratio grows large.

![Straggler and bad node](./straggler_and_bad_node.svg)

### Debug / fix

- Per-rank tokens/step + forward ms.
- Length bucketing / token-budget batching.
- Do not “tune NCCL” first.

---

## 6. Bad node — persistent host preprocess skew

True HBM/thermal faults need DCGM/Xid. What we **can** reproduce on Modal is the other common “bad node” shape in multimodal / tokenization-heavy jobs: **one rank’s host path is permanently expensive**, GPU kernels look fine, pre-collective local timers scream.

**Lab:** rank 0 does heavy host tokenize/histogram work every step; peer does minimal work → local timer ratio flags rank 0.

### Debug / fix

- Split **host ms vs GPU ms**.
- If GPU kernels are fine and host is not → CPU affinity / decoder / noisy neighbor.
- If GPU phase is slow and stable on one UUID → cordon the node.

---

## 7. NCCL hang — empty microbatch `continue`

The classic desync:

```python
if valid_tokens == 0:
    continue  # skips DDP forward/backward → peers hang in allreduce
```

![Hang and throughput cliff](./hang_and_throughput_cliff.svg)

**Lab:** rank 1 hits an all-pad pack, `continue`s; rank 0 blocks inside DDP/NCCL until watchdog kill / peer-closed error.

### Debug / fix

- Identical control flow into every collective — empty packs still need a noop participation or a collective-safe skip protocol.
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, heartbeats with last completed step per rank.

---

## 8. Throughput cliffs — tokens/rank vs real grad sync

No fake `all_reduce(junk)`. Sweep tokens/rank on `TinyTransformerLM`; communication is the **real DDP gradient allreduce**. Tiny packs fall off a cliff.

### Debug / fix

- Plot tokens/s vs tokens/rank.
- Raise microbatch / seqlen; overlap comm; stop cosplaying large `world_size` with 8-token packs.

---

## Runbook cheat sheet

| If you see… | First probe | Likely class | First fix |
|---|---|---|---|
| `nan`/`inf` | which recipe class + first rank | Numerics | scaler / mask / skip Adam step |
| rare loss explosions | CE vs aux separately + LR | Aux/schedule/data | fix aux weight / schedule step |
| fine loss, bad eval / resume | param checksum across ranks | Silent drift | never rank-gate clip/EMA |
| climbing CUDA alloc | alloc curve | Leak | stop retaining step tensors |
| all ranks slow, one long fwd | per-rank tokens + fwd ms | Straggler | bucket / token budget |
| one rank host-heavy forever | host vs GPU timers + DCGM | Bad node / host skew | cordon or fix CPU path |
| stuck in NCCL | last step per rank + control flow | Empty-batch skip | remove `continue` past DDP |
| tiny packs, terrible scaling | tokens/s vs tokens/rank | Cliff | enlarge local work |

## Reproduce

```bash
uv run modal run playground/dist_failure_modal.py
uv run modal run playground/dist_failure_modal.py --case numerical_drift
uv run python playground/dist_failure_modal.py --case nccl_hang
```

## Takeaways

1. **Reproduce with the real abstraction** (Transformer LM + DDP), not cartoons.
2. **Loss lies** — drift and empty-batch hangs keep CE looking fine until they do not.
3. **Split host / forward / collective timers** or you will mis-blame NCCL.
4. **`if is_main:` around clip/EMA/step is a desync bug**, not a cleanup.
5. **Empty packs must still participate in collectives.**

Committed numbers: Modal **2×A10G + NCCL**. Same code runs CPU/`gloo` locally when you only need the failure shape.
