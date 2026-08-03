---
title: "ZeRO: Partitioning Optimizer State, Gradients, and Parameters"
date: 2025-04-27
tags: ["LLM", "Distributed Training", "ZeRO", "DeepSpeed", "Memory", "Optimizer"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: zero_stages.svg
  alt: "ZeRO stages partition optimizer state, gradients, and parameters"
  relative: true
---

# ZeRO: Partitioning Optimizer State, Gradients, and Parameters

Plain DDP is clean but wasteful. Every data-parallel rank stores the same parameters, gradients, fp32 master weights, and optimizer moments. At small scale that redundancy is convenient. At LLM scale it is the memory wall.

ZeRO, the Zero Redundancy Optimizer, keeps data-parallel semantics and removes the redundant storage one category at a time. ZeRO-1 shards optimizer state. ZeRO-2 shards optimizer state and gradients. ZeRO-3 shards optimizer state, gradients, and parameters ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)).

The mechanism is not mystical. If a rank does not need a byte right now, do not store that byte there. Gather it just in time, reduce it to the owner, then release it.

## TL;DR

- Training memory splits into **model states** and **residual states**. ZeRO first attacks model-state redundancy.
- Mixed precision Adam commonly needs about `16 * Psi` bytes of static model state for `Psi` parameters before activations.
- ZeRO-1 partitions optimizer state. ZeRO-2 also partitions gradients. ZeRO-3 also partitions parameters.
- ZeRO has model-sharded storage but data-parallel semantics: each rank still trains on different data.
- The cost is communication and runtime discipline: bucket, prefetch, release, and overlap.
- ZeRO-Offload moves optimizer state and compute to CPU DRAM ([arXiv:2101.06840](https://arxiv.org/abs/2101.06840)). ZeRO-Infinity extends the hierarchy to NVMe with prefetch and tiling ([arXiv:2104.07857](https://arxiv.org/abs/2104.07857)).
- PyTorch FSDP is the same broad family: shard parameters and gather them around module execution.
- Reproducible figures for this post: [`playground/llm_training_series_figures.py`](https://github.com/duoan/duoan.github.io/blob/main/playground/llm_training_series_figures.py).

## 1. Name the Memory

Before optimizing memory, split it into the right buckets.

![Training memory includes model states and residual states](memory_breakdown.svg)

**Model states** scale directly with parameter count:

- Parameters used by forward and backward.
- Gradients produced by backward.
- Optimizer state, such as Adam's first and second moments.
- fp32 master parameters in many mixed precision optimizers.

**Residual states** come from execution:

- Activations needed for backward.
- Temporary communication buffers.
- Kernel workspaces.
- Fragmentation inside the allocator.

The original ZeRO paper makes this distinction explicit because the remedies differ ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)). Sharding optimizer state does not remove activation memory. Activation checkpointing does not remove Adam moments. You need to know which wall you are hitting.

## 2. The 16 Bytes per Parameter Problem

Let `Psi` be parameter count. A common mixed precision Adam layout is:

```text
fp16/bf16 parameters: 2 * Psi bytes
fp16/bf16 gradients:  2 * Psi bytes
fp32 master weights:  4 * Psi bytes
Adam first moment:    4 * Psi bytes
Adam second moment:   4 * Psi bytes
```

Total:

```text
16 * Psi bytes
```

![Mixed precision still stores fp32 master weights and Adam moments](mixed_precision_memory.svg)

For a 10B parameter model, that is about 160 GB of model-state memory before activations. Plain DDP puts those 160 GB on every data-parallel rank.

The exact budget changes with optimizer and dtype. SGD stores less. Adafactor changes the moment structure. Some bf16 stacks avoid fp32 master weights. The shape remains: optimizer state is large, and data parallelism replicates it.

## 3. The ZeRO Principle

Data parallelism requires every rank to contribute gradients for its data shard and apply the same logical update. It does **not** require every rank to permanently store every optimizer byte.

ZeRO assigns ownership over shards:

```text
rank 0 owns shard 0
rank 1 owns shard 1
...
rank N-1 owns shard N-1
```

When a full tensor is needed, ranks communicate. When a rank only needs its owned shard, it keeps only that shard.

![ZeRO stages progressively remove redundancy](zero_stages.svg)

That is the entire trade: lower HBM footprint in exchange for more collective communication and stricter tensor lifetimes.

## 4. ZeRO-1: Shard Optimizer State

ZeRO-1 partitions optimizer state across data-parallel ranks ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)).

Each rank keeps:

- Full fp16/bf16 parameters.
- Full gradients, or full gradient buckets during reduction.
- Only `1/N` of optimizer state and owned master weights, depending on implementation.

After backward, gradients are reduced. Each rank updates the parameter shard it owns. Updated parameter shards are then gathered so every rank has full parameters for the next forward pass.

Why this is a large first win: in the 16-byte estimate, fp32 master weights plus Adam moments are 12 bytes per parameter. Sharding that across `N` ranks removes the largest replicated block.

Communication is still close to familiar data parallelism. Practical implementations use reduce-scatter and all-gather patterns rather than literally materializing every full tensor in every phase.

Use ZeRO-1 when the model almost fits and optimizer state is the main problem.

## 5. ZeRO-2: Shard Gradients Too

ZeRO-2 partitions optimizer state and gradients ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)).

Parameters remain fully resident during forward and backward. Gradients do not. After backward, the gradient reduction naturally becomes reduce-scatter:

```text
local gradients on all ranks
    -> reduce-scatter
reduced gradient shard on owning rank
    -> local optimizer update
    -> all-gather updated parameter shards
```

This removes another replicated tensor. For dense training, ZeRO-2 is often a strong default because it gives a large memory drop while keeping parameter access simple during forward/backward.

The communication shape should look familiar from [ring all-reduce](../data-parallelism-ddp-ring-allreduce/): all-reduce is reduce-scatter plus all-gather. ZeRO-2 changes which pieces stay materialized and where the optimizer runs.

Use ZeRO-2 when gradients are a meaningful memory consumer and you do not need to shard parameters during compute.

## 6. ZeRO-3: Shard Parameters

ZeRO-3 partitions optimizer state, gradients, and parameters ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)).

No rank permanently stores a full model replica. Execution becomes:

1. Before a module runs, all-gather the needed parameter shards.
2. Run forward or backward with the full parameter for that module.
3. Release non-owned parameter shards as soon as their lifetime ends.
4. Reduce-scatter gradients back to owning ranks.
5. Update only local owned shards.

This is the stage that breaks the "one full replica per DP rank" memory assumption.

It is also the stage most sensitive to implementation quality. Naive ZeRO-3 can turn a training step into a long chain of blocking all-gathers. Good ZeRO-3 depends on:

- Large buckets, not tiny per-parameter collectives.
- Prefetch of upcoming module parameters.
- Immediate release after use.
- Overlap with compute where dependencies allow it.
- Topology-aware groups.
- Checkpoint formats that understand sharded state.

ZeRO-3 is powerful when full parameters cannot remain resident. It is overkill when ZeRO-2 already fits and the job is bandwidth-bound.

## 7. ZeRO Is Not Tensor Parallelism

This distinction prevents many debugging mistakes.

![ZeRO and model parallelism split different responsibilities](zero_vs_model_parallel.svg)

In tensor parallelism, ranks split the layer computation. One rank owns columns, another owns rows, or heads are split across ranks. Activations flow through a distributed computation graph.

In ZeRO-3, ranks split persistent storage. When a module executes, the needed full parameter is reconstructed for that module. Each rank still processes a different data shard.

That is why ZeRO composes naturally with [Megatron tensor parallelism](../tensor-parallelism-megatron/). TP reduces per-rank layer compute and parameter slices inside a TP group. ZeRO reduces redundant model-state storage across data-parallel replicas.

## 8. Residual States: ZeRO-R

The original ZeRO work also discusses residual-state optimizations under ZeRO-R ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)).

**Partitioned activation checkpointing** shards activation checkpoints across model-parallel ranks and gathers them for recomputation. This matters when long sequence length makes activations rival model states.

**Constant-size buffers** keep communication temporary memory predictable. As rank count grows, shards get small; collecting them into stable buffers avoids a mess of tiny messages and allocator churn.

**Memory defragmentation** addresses allocator reality. A job can have enough total free memory but fail to allocate one contiguous block. Reusing and compacting buffers reduces that failure mode.

These are less famous than ZeRO-1/2/3, but they decide whether large jobs run for days without allocator surprises.

## 9. Offload: CPU DRAM Is a Slower Memory Tier

ZeRO-Offload moves optimizer state and optimizer computation to CPU for some configurations ([arXiv:2101.06840](https://arxiv.org/abs/2101.06840)).

![ZeRO-Offload and Infinity stage model states through CPU and NVMe](zero_offload.svg)

The hierarchy becomes:

```text
GPU HBM  -> fastest, smallest
CPU DRAM -> larger, slower, connected by PCIe/NVLink-C2C depending on system
NVMe     -> much larger, much slower
```

Offload helps only if GPU compute can overlap with host transfers and CPU optimizer work. If the GPU waits on PCIe every layer, the memory win becomes a throughput loss.

The practical use case is capacity-constrained training where peak throughput is not the only target: fine-tuning, smaller clusters, or runs where HBM is the hard blocker and CPU memory is available.

## 10. Infinity: NVMe Joins the Schedule

ZeRO-Infinity generalizes offload by staging parameters, gradients, and optimizer states across GPU, CPU, and NVMe ([arXiv:2104.07857](https://arxiv.org/abs/2104.07857)).

NVMe is not "more GPU memory." It is a storage tier that must be streamed. The runtime needs:

- Tiling so working sets fit in HBM.
- Prefetch far enough ahead to hide NVMe and CPU latency.
- Eviction of cold shards.
- Large sequential transfers instead of random small reads.
- Overlap between compute, CPU memory movement, and NVMe IO.

Infinity is a scheduling system around a memory hierarchy. Treat it as such. The failure mode is enabling offload and discovering that the GPU is now a very expensive device waiting on storage.

## 11. FSDP as the Same Family

PyTorch Fully Sharded Data Parallel (FSDP) lives in the same design family as ZeRO-3: shard parameters across data-parallel ranks, all-gather before module execution, reduce-scatter gradients, and free full parameters after use.

The API and implementation details differ from DeepSpeed ZeRO, but the mental model transfers:

- Parameter shards are persistent.
- Full parameters are temporary.
- Bucket size, prefetch, wrapping policy, and overlap decide performance.
- Checkpointing must understand sharded state.

Use this framing when comparing DeepSpeed ZeRO and PyTorch FSDP. The important question is not the brand. It is tensor lifetime and communication placement.

## 12. Code

Read these code paths for the actual contracts:

- [DeepSpeed ZeRO runtime](https://github.com/deepspeedai/DeepSpeed/tree/master/deepspeed/runtime/zero): partitioning, parameter coordination, offload, gradient reduction, and optimizer state management.
- [DeepSpeed ZeRO stage 3](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/stage3.py): all-gather/release behavior and stage-3 orchestration.
- [DeepSpeed ZeRO-Offload](https://github.com/deepspeedai/DeepSpeed/tree/master/deepspeed/runtime/zero): CPU optimizer/offload paths live under the same runtime tree.
- [PyTorch FSDP](https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp): related sharded data-parallel implementation.
- [Megatron-LM distributed optimizer](https://github.com/NVIDIA/Megatron-LM): production distributed optimizer patterns that compose with TP/PP.

The useful exercise is to trace a parameter through forward prefetch, use, release, backward gather, gradient reduce-scatter, and optimizer update.

## 13. Practical Guidance

Use the lowest stage that solves the memory problem:

1. **Plain DDP**: when a full replica fits and communication is acceptable.
2. **ZeRO-1**: optimizer state is the blocker.
3. **ZeRO-2**: gradients are also a major memory block.
4. **ZeRO-3 / FSDP**: parameters cannot stay fully resident.
5. **Offload / Infinity**: HBM is still the blocker and the throughput target can tolerate careful staging.

Add activation checkpointing independently. ZeRO reduces model-state memory; it does not make long-context activations disappear.

The most common mistake is enabling the most aggressive stage first. ZeRO-3 can unlock scale, but it also adds more collectives, more scheduling state, and more checkpoint complexity. The best configuration is the simplest one that fits and keeps GPUs busy.

## References

- Samyam Rajbhandari et al., [*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054), SC 2020.
- Jie Ren et al., [*ZeRO-Offload: Democratizing Billion-Scale Model Training*](https://arxiv.org/abs/2101.06840), USENIX ATC 2021.
- Samyam Rajbhandari et al., [*ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning*](https://arxiv.org/abs/2104.07857), SC 2021.
- Yanping Huang et al., [*GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*](https://arxiv.org/abs/1811.06965), NeurIPS 2019.
- Mohammad Shoeybi et al., [*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*](https://arxiv.org/abs/1909.08053), 2019.
- Code: [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), [`deepspeed/runtime/zero/`](https://github.com/deepspeedai/DeepSpeed/tree/master/deepspeed/runtime/zero), [PyTorch FSDP](https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp), [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
