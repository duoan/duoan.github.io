---
title: "Megatron Internals I: Building the DP / TP / PP Process Groups"
date: 2025-04-12
tags: ["LLM", "Megatron", "Distributed Training", "Data Parallel", "Tensor Parallel", "Pipeline Parallel"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: process_group_mesh.svg
  alt: "Megatron process groups carved from a DP, PP, and TP rank mesh"
  relative: true
---

# Megatron Internals I: Building the DP / TP / PP Process Groups

Megatron-LM's first trick is not tensor-parallel matmul.
It is rank bookkeeping.
Before the model runs, every process must know the small set of peers it will communicate with for tensor parallelism, pipeline parallelism, data parallelism, embeddings, and optimizer state.

The original [Megatron-LM paper](https://arxiv.org/abs/1909.08053) introduced intra-layer tensor parallelism for transformer training.
The later [Megatron-LM scaling paper](https://arxiv.org/abs/2104.04473) put tensor parallelism, pipeline parallelism, and data parallelism into one training system.
This post is the control plane underneath that system: how a flat list of ranks becomes a DP / PP / TP mesh.

## TL;DR

- Megatron starts with one process per GPU and one global `torch.distributed` world group.
- It then interprets `world_size` as a regular mesh: `world_size = dp_size * pp_size * tp_size`.
- A **TP group** fixes `(dp, pp)` and varies `tp`; those ranks cooperate inside one layer.
- A **PP group** fixes `(dp, tp)` and varies `pp`; those ranks pass activations between pipeline stages.
- A **DP group** fixes `(pp, tp)` and varies `dp`; those ranks own equivalent shards and synchronize gradients.
- The same rank belongs to several groups at once. The group handle tells later code which peers matter for the current operation.
- Seed handling is part of initialization: DP replicas should match, while TP and PP shards should not accidentally duplicate each other.
- For a broader tensor-parallel overview, see [Tensor Parallelism in Megatron](../tensor-parallelism-megatron/). For optimizer-state sharding, see [ZeRO Redundancy Optimizer](../zero-redundancy-optimizer/).

## 1. The question initialization answers

Distributed training launches the same Python program many times.
The program can inspect `rank`, `world_size`, and `local_rank`, but it does not start with a semantic map.
Megatron initialization creates that map.

Each rank needs to answer five questions:

1. Which CUDA device do I bind to?
2. Which ranks share a tensor-parallel layer with me?
3. Which ranks are before and after me in the pipeline?
4. Which ranks own the same parameter shard in another data-parallel replica?
5. Which random-number streams should be shared, and which should differ?

If any answer is wrong, the failure mode is usually ugly.
A bad group can hang in `all_reduce`.
A bad pipeline neighbor can send tensors of the wrong shape.
A bad seed can silently create duplicate parameter shards.
This is why mature distributed-training code prints rank maps at startup.
That log is not decoration; it is a correctness artifact.

## 2. Rank mesh first, process groups second

Think of the global rank list as a 3D tensor:

```text
world_size = dp_size * pp_size * tp_size
rank       = rank_of(dp_rank, pp_rank, tp_rank)
```

The exact flattening order is implementation detail.
The invariant is not.
Every global rank has one coordinate on each axis.

![World ranks as a DP x PP x TP mesh](process_group_mesh.svg)

Once you have the mesh, process groups are slices:

- **Tensor parallel:** keep `dp_rank` and `pp_rank` fixed, vary `tp_rank`.
- **Pipeline parallel:** keep `dp_rank` and `tp_rank` fixed, vary `pp_rank`.
- **Data parallel:** keep `pp_rank` and `tp_rank` fixed, vary `dp_rank`.

This is the clean mental model.
The code adds extra groups for embeddings, position embeddings, sequence parallelism, context parallelism, and sometimes expert parallelism.
But the construction is still "rank lists first, handles second."

## 3. A concrete 16-rank example

Suppose the job has:

```text
world_size = 16
tp_size    = 2
pp_size    = 4
dp_size    = 2
```

One full model replica consumes `tp_size * pp_size = 8` GPUs.
With 16 GPUs, the job can run two data-parallel replicas.

![The same ranks sliced into TP, PP, and DP groups](dp_tp_pp_ranks.svg)

For rank 0, typical groups look like:

- TP group: ranks `[0, 1]`, same data replica and same pipeline stage.
- PP group: ranks `[0, 4, 8, 12]`, same data replica and same TP shard across stages.
- DP group: ranks `[0, 2]`, same model shard in different data replicas.

These are not new processes.
They are subgroup handles over the same global processes.
The same GPU can all-reduce with rank 1 during attention, send an activation to rank 4 at a stage boundary, and reduce a gradient with rank 2 before the optimizer step.

## 4. What PyTorch gives Megatron

PyTorch creates the global rendezvous:

```python
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(
    backend="nccl",
    init_method=init_method,
    world_size=world_size,
    rank=rank,
)
```

That world group is necessary, but it is too blunt.
It can say "all ranks exist."
It cannot say "these two ranks own one sharded MLP weight."
Megatron adds the semantic groups on top:

```python
for dp in range(dp_size):
    for pp in range(pp_size):
        ranks = [rank_of(dp, pp, tp) for tp in range(tp_size)]
        tp_group = dist.new_group(ranks)

for dp in range(dp_size):
    for tp in range(tp_size):
        ranks = [rank_of(dp, pp, tp) for pp in range(pp_size)]
        pp_group = dist.new_group(ranks)

for pp in range(pp_size):
    for tp in range(tp_size):
        ranks = [rank_of(dp, pp, tp) for dp in range(dp_size)]
        dp_group = dist.new_group(ranks)
```

The important engineering rule: every rank must call group creation in the same deterministic order.
Ranks only keep the handles they belong to, but the creation calls have to line up globally.

## 5. What each axis moves

The axes exist because different tensors move at different times.

**Tensor parallelism moves layer intermediates.**
Column-parallel and row-parallel linear layers split matrix multiplies across ranks.
They need all-gather or all-reduce collectives inside transformer blocks.
That is the subject of [Megatron Internals II](../megatron-model-parallel-internals/) and the broader [tensor-parallelism post](../tensor-parallelism-megatron/).

**Pipeline parallelism moves activations.**
One stage owns early layers, another owns middle layers, and another owns late layers.
Forward sends activation tensors downstream.
Backward sends activation gradients upstream.
The collective is usually point-to-point, but it is latency-sensitive because pipeline bubbles waste whole stages.
GPipe gives the clean baseline model for this scheduling problem in [its paper](https://arxiv.org/abs/1811.06965).

**Data parallelism moves gradients.**
After backward, ranks that own equivalent parameter shards average gradients.
The DP axis is also where optimizer-state sharding methods such as [ZeRO](https://arxiv.org/abs/1910.02054) partition redundant state.
For the practical memory breakdown, see [ZeRO Redundancy Optimizer](../zero-redundancy-optimizer/).

## 6. Topology is not optional

The rank mesh is logical.
The cluster fabric is physical.
If the logical mesh ignores the fabric, the job can be correct and still slow.

The usual rule is simple:

- keep TP on the fastest links, usually inside an NVLink island or one high-bandwidth node;
- let DP cross nodes when gradient accumulation and bucket sizes can hide the cost;
- use PP when layer partitioning reduces memory or avoids worse cross-node TP traffic.

That rule follows the communication frequency.
TP collectives sit inside almost every transformer block.
DP gradient reductions happen once per bucket per step.
PP sends at stage boundaries.
The tensor that moves most often deserves the best link.

This is also why "just increase TP" is not a plan.
TP reduces per-rank parameter and activation slices, but it adds layer-critical collectives.
Once those collectives cross slow links, training can get worse even if memory improves.

## 7. Seeds are distributed state

Randomness is part of the parallel contract.
DP replicas should initialize the same shard and apply equivalent dropout for equivalent tensors.
TP ranks often own different slices of the same weight.
PP ranks own different layers.

A useful invariant:

```text
same DP coordinate for the same model shard -> same parameter stream
different TP or PP shard                       -> different parameter stream
checkpoint recomputation                       -> replay the same dropout stream
```

Megatron uses model-parallel RNG tracking so activation checkpointing can recompute forward passes during backward without changing dropout masks.
This is not just reproducibility.
It is mathematical correctness.
If recomputation draws a different dropout mask, the backward pass is no longer the gradient of the forward pass that produced the loss.

## 8. Where memory systems hook in

Initialization is not the optimizer.
It gives the optimizer enough topology to be correct.

ZeRO-style optimizer sharding is defined over DP redundancy.
Tensor parallelism and pipeline parallelism define model ownership.
That distinction matters:

- a TP rank owns a slice of a layer and needs model-parallel collectives for dense math;
- a DP rank owns an equivalent slice in another replica and needs gradient synchronization;
- a ZeRO optimizer shard owns a partition of optimizer state and needs DP-aware gather or reduce-scatter behavior.

Mixed precision adds another layer.
The optimizer must decide overflow and clipping consistently across all ranks that form one logical model update.
That is why [Megatron Internals III](../megatron-mixed-precision-training/) starts from this topology.

## 9. Initialization flow

![Megatron distributed initialization flow](init_flow.svg)

A practical startup sequence is:

1. launch one process per GPU;
2. parse rank, world size, local rank, TP size, PP size, and optional extra axes;
3. bind the process to its local CUDA device;
4. initialize the global `torch.distributed` process group;
5. create TP, PP, DP, embedding, and model-parallel groups from deterministic rank lists;
6. initialize model-parallel RNG streams;
7. build model modules that call into those group handles;
8. build optimizer and distributed memory hooks that rely on the same topology.

Notice the direction of dependency.
Layer code should not rediscover rank topology.
It should ask the parallel-state module for "my tensor-model-parallel group" and perform the local collective.

## 10. Failure modes worth recognizing

Most startup bugs are boring and expensive:

- **Different group creation order:** ranks call `new_group` with different rank lists or in a different order, and later collectives hang.
- **Wrong local device binding:** two processes land on one GPU, so memory and NCCL behavior look strange.
- **Non-divisible world size:** `world_size` is not divisible by the required parallel axes, so the mesh cannot tile.
- **Topology-blind TP:** a legal TP group crosses slow links and dominates step time.
- **Seed coupling:** TP shards initialize identically and reduce model capacity without crashing.

Log the groups, check divisibility, and check device binding. For large runs, those checks are cheaper than one failed hour on the cluster.

## Code

- Megatron startup and distributed initialization: [`megatron/training/initialize.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/initialize.py).
- Megatron process-group construction and rank helpers: [`megatron/core/parallel_state.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py).
- Tensor-parallel callers that depend on the TP group: [`megatron/core/tensor_parallel/`](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/tensor_parallel).
- Optimizer code that depends on DP and model-parallel ownership: [`megatron/core/optimizer/`](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/optimizer).
- DeepSpeed ZeRO implementation for the DP-side memory story: [`deepspeed/runtime/zero/`](https://github.com/deepspeedai/DeepSpeed/tree/master/deepspeed/runtime/zero).

## References

- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), 2019.
- Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), 2021.
- Huang et al., [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965), NeurIPS 2019.
- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), SC 2020.
