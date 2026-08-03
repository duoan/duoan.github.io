---
title: "LLM Training Series: A Systems Map for Distributed Transformers"
date: 2025-03-02
tags: ["LLM", "Distributed Training", "Systems", "Parallelism", "Megatron-LM", "DeepSpeed"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: series_map.svg
  alt: "Map of distributed LLM training techniques across DP, TP, PP, SP, ZeRO, and MoE"
  relative: true
---

# LLM Training Series: A Systems Map for Distributed Transformers

LLM training stops being "run the model on more GPUs" the moment one replica no longer fits, one link becomes the step time, or one schedule leaves half the cluster idle. From that point on, the training run is a placement problem.

The stable question is simple: **which bytes are replicated, which bytes are sharded, and which link moves them on the critical path?** Data parallelism, tensor parallelism, pipeline parallelism, ZeRO, sequence/context parallelism, and expert parallelism are different answers to that question. The public systems that matter - GPipe, Megatron-LM, DeepSpeed ZeRO, PyTorch DDP, GShard, Switch Transformer, and modern Megatron-Core - all expose the same constraints in different shapes ([GPipe](https://arxiv.org/abs/1811.06965), [Megatron-LM](https://arxiv.org/abs/1909.08053), [ZeRO](https://arxiv.org/abs/1910.02054), [PyTorch Distributed](https://arxiv.org/abs/2006.15704), [GShard](https://arxiv.org/abs/2006.16668), [Switch Transformer](https://arxiv.org/abs/2101.03961)).

![Distributed LLM training techniques form a hybrid parallelism map](series_map.svg)

## TL;DR

- **Data parallelism** replicates the model and shards the batch. The hard part is reducing gradients without turning one server or one slow rank into the step time.
- **Tensor parallelism** splits the algebra inside a Transformer block. It moves activation-sized tensors every layer, so it belongs on fast local links.
- **Pipeline parallelism** splits the layer stack. It buys capacity and another scale axis, then charges you in bubbles, micro-batches, and schedule complexity.
- **ZeRO/FSDP-style sharding** keeps data-parallel semantics while partitioning optimizer state, gradients, and parameters.
- **Sequence/context parallelism** appears when long context makes activation and attention memory larger than the model-state problem.
- **MoE expert parallelism** shards experts and routes tokens. Sparse compute reduces FLOPs per token, but all-to-all, load balance, and small GEMMs become first-class.
- The production answer is hybrid. Single-axis scaling is a warmup exercise, not a large-model recipe.

## Reading Order

Read these in order if you want the stack to build cleanly. Each post still stands alone.

1. [Pipeline Parallelism from First Principles: Why GPipe Split the Batch](../pipeline-parallelism-gpipe/)
2. [Data Parallelism: From Parameter Server to Ring All-Reduce](../data-parallelism-ddp-ring-allreduce/)
3. [ZeRO: Partitioning Optimizer State, Gradients, and Parameters](../zero-redundancy-optimizer/)
4. [Tensor Parallelism in Megatron-LM: Splitting Layers, Not Stacks](../tensor-parallelism-megatron/)
5. [Megatron Internals I: Building the DP / TP / PP Process Groups](../megatron-distributed-init/)
6. [Megatron Internals II: Column/Row Parallel Linear and Vocab Parallel Embedding](../megatron-model-parallel-internals/)
7. [Megatron Internals III: Mixed Precision, Loss Scaling, and Grad Clipping](../megatron-mixed-precision-training/)
8. [MoE Parallelism Principles: GShard, Expert Parallel, and All-to-All](../moe-expert-parallelism-principles/)
9. [MoE Internals: DeepSpeed-Megatron Expert Parallel Implementation](../moe-deepspeed-megatron-internals/)
10. [Sequence Parallelism I: Megatron SP](../sequence-parallelism-megatron-sp/)
11. [Sequence Parallelism II: DeepSpeed Ulysses](../sequence-parallelism-ulysses/)
12. [Sequence Parallelism III: Ring Attention](../ring-attention/)
13. [Sequence Parallelism IV: Megatron Context Parallel](../megatron-context-parallel/)
14. [Hiding Tensor-Parallel Collectives: AG/RS Overlap in Megatron](../megatron-tp-comm-overlap/)
15. [The ZeRO-3 Diagram Most People Remember Is Wrong](../zero3-intra-layer-partitioning/)

## The Map

Start by naming the partition. Most confusion in distributed training comes from mixing these rows.

| Technique | Partitioned thing | Hot communication | Why it exists |
|---|---|---|---|
| Data parallelism | Batch | Gradient all-reduce / reduce-scatter | Throughput when one replica fits |
| Tensor parallelism | Matrices, heads, vocab shards | Activation all-reduce / all-gather | One layer is too large or too slow |
| Pipeline parallelism | Layer stack | Activation send/recv | The stack does not fit, or scale needs another axis |
| ZeRO / FSDP | Optimizer state, gradients, parameters | Reduce-scatter / all-gather | Data-parallel replicas waste model-state memory |
| Sequence parallelism | Sequence activations | All-gather / reduce-scatter | Long context makes activations dominate |
| Context / ring attention | K/V context blocks | Ring exchange of attention blocks | Full-context attention does not fit |
| Expert parallelism | Experts and routed tokens | All-to-all | Sparse models have many inactive parameters |

These are not competing features. They compose because they attack different tensors.

A dense 70B-class run may use tensor parallelism inside an NVSwitch domain, data parallelism across replicas, ZeRO or a distributed optimizer across the DP dimension, and pipeline parallelism only if depth still does not fit. A long-context run adds sequence or context parallelism. An MoE run adds expert parallelism, and suddenly all-to-all placement matters as much as GEMM throughput.

## Canonical Axes

### Data parallelism

The invariant is that every rank applies the same update. Parameter servers made that explicit by centralizing gradient aggregation, but they concentrate network traffic at the server ([Scaling Distributed Machine Learning with the Parameter Server](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu)). Ring all-reduce removes that hot spot by decomposing all-reduce into reduce-scatter plus all-gather; Baidu popularized the pattern for deep learning clusters ([arXiv:1702.05847](https://arxiv.org/abs/1702.05847)), Horovod made it easy to use ([arXiv:1802.05799](https://arxiv.org/abs/1802.05799)), and PyTorch DDP wraps the same idea around autograd buckets ([arXiv:2006.15704](https://arxiv.org/abs/2006.15704)).

### Pipeline parallelism

Pipeline parallelism treats the model as a chain. GPipe's key mechanism is micro-batching: split one mini-batch into `M` pieces so `K` pipeline stages do useful work instead of waiting through a `(K - 1) / K` bubble ([arXiv:1811.06965](https://arxiv.org/abs/1811.06965)). PipeDream then showed what changes when the schedule is asynchronous and weight versions can be stale ([arXiv:1806.03377](https://arxiv.org/abs/1806.03377)). Megatron-LM's 1F1B and interleaved schedules are the production descendants for dense Transformer training ([arXiv:2104.04473](https://arxiv.org/abs/2104.04473)).

### ZeRO and sharded data parallel

ZeRO starts from one fact: Adam state, gradients, master weights, and parameters are replicated across data-parallel ranks even though each rank only needs some of them at a given moment. ZeRO-1 shards optimizer state, ZeRO-2 also shards gradients, and ZeRO-3 shards parameters too ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)). ZeRO-Offload and ZeRO-Infinity extend the storage hierarchy to CPU DRAM and NVMe, which helps only when prefetch and overlap keep the GPU fed ([arXiv:2101.06840](https://arxiv.org/abs/2101.06840), [arXiv:2104.07857](https://arxiv.org/abs/2104.07857)).

### Tensor parallelism

Megatron-LM's tensor parallelism is not arbitrary matrix slicing. It uses a column-parallel first MLP projection, local GeLU, and a row-parallel second projection so communication lands after the nonlinearity, not before it ([arXiv:1909.08053](https://arxiv.org/abs/1909.08053)). Attention is split by heads. Vocab embeddings and logits are sharded by token range. The later Megatron paper shows how this TP axis composes with DP and PP at cluster scale ([arXiv:2104.04473](https://arxiv.org/abs/2104.04473)).

### Sequence, context, and MoE

Long context shifts the bottleneck from model state to activations and attention. Sequence parallelism shards sequence-dimension activations; ring attention shards the context blocks and circulates K/V blocks instead of materializing the full attention problem on every rank. MoE shifts the problem again: GShard and Switch Transformer show that sparse experts can scale parameter count without dense FLOPs, but the system pays in routing, expert placement, all-to-all, and load balance ([GShard](https://arxiv.org/abs/2006.16668), [Switch Transformer](https://arxiv.org/abs/2101.03961)).

## Code

The papers are useful, but the contracts are clearest in code:

- [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and Megatron-Core: tensor parallel layers, pipeline schedules, distributed optimizer, sequence/context parallelism, and MoE.
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), especially [`deepspeed/runtime/zero/`](https://github.com/deepspeedai/DeepSpeed/tree/master/deepspeed/runtime/zero): ZeRO stages, offload, partitioning, and optimizer-state orchestration.
- [PyTorch distributed](https://github.com/pytorch/pytorch/tree/main/torch/distributed) and [`DistributedDataParallel`](https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py): process groups, c10d collectives, autograd hooks, and gradient buckets.
- [PyTorch pipelining](https://github.com/pytorch/pytorch/tree/main/torch/distributed/pipelining): the modern PyTorch lineage for pipeline schedules.
- [Horovod](https://github.com/horovod/horovod): a compact reference for ring-allreduce-centered data parallelism across frameworks.

## The Recurring Checks

Before choosing a parallel recipe, answer these in this order:

1. **What must be resident?** Parameters, gradients, optimizer state, activations, KV/cache-like temporaries, communication buffers.
2. **What can be sharded without changing semantics?** Model states are easier than activations; activations are easier than arbitrary dynamic routing.
3. **Which collective moves the hot bytes?** All-reduce, reduce-scatter, all-gather, all-to-all, send/recv, or a custom ring.
4. **Can communication overlap compute?** Exposed communication is the cost that matters.
5. **Which physical link carries it?** NVLink/NVSwitch, PCIe, InfiniBand, Ethernet, CPU DRAM, and NVMe are not interchangeable.
6. **What new failure mode appears?** Bubbles, stale weights, small collectives, load imbalance, memory fragmentation, checkpoint complexity, or graph breaks.

## Practical Order

The least painful plan usually looks like this:

1. Estimate model-state memory and activation memory.
2. Use plain data parallelism while one replica fits.
3. Add ZeRO/FSDP when replicated model states are the blocker.
4. Add tensor parallelism when individual layers are too wide or too slow.
5. Add sequence/context parallelism when long context dominates activations or attention.
6. Add pipeline parallelism when depth still does not fit or the cluster needs another scale axis.
7. Add expert parallelism only when the model architecture is sparse.

This is not a law. It is a bias toward the smallest system that fits the constraint in front of you.

## Related Posts Already on This Blog

- [From Scaling Laws to Cluster Size](../large-model-capacity-plan/) - tokens, FLOPs, and GPU-hours before the parallel recipe.
- [Large MoE Performance: The Three Walls After Sparsity](../large-moe-from-sparsity-to-communication/) - production MegaScale / Megatron-Core MoE co-design.
- [Learning PyTorch DDP Performance Tuning on a One-GPU Machine](../learning-ddp-performance-tuning-on-one-gpu/) - DDP pathologies you can reproduce locally.
- [Why Variable Sequence Length Breaks DDP Throughput](../why-variable-sequence-length-breaks-ddp-throughput/) - token imbalance under data parallelism.

## References

- Yanping Huang et al., [*GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*](https://arxiv.org/abs/1811.06965), NeurIPS 2019.
- Aaron Harlap et al., [*PipeDream: Fast and Efficient Pipeline Parallel DNN Training*](https://arxiv.org/abs/1806.03377), SOSP 2019.
- Mohammad Shoeybi et al., [*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*](https://arxiv.org/abs/1909.08053), 2019.
- Deepak Narayanan et al., [*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*](https://arxiv.org/abs/2104.04473), SC 2021.
- Samyam Rajbhandari et al., [*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054), SC 2020.
- Jie Ren et al., [*ZeRO-Offload: Democratizing Billion-Scale Model Training*](https://arxiv.org/abs/2101.06840), USENIX ATC 2021.
- Samyam Rajbhandari et al., [*ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning*](https://arxiv.org/abs/2104.07857), SC 2021.
- Mu Li et al., [*Scaling Distributed Machine Learning with the Parameter Server*](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu), OSDI 2014.
- Andrew Gibiansky et al., [*Bringing HPC Techniques to Deep Learning*](https://arxiv.org/abs/1702.05847), 2017.
- Alexander Sergeev and Mike Del Balso, [*Horovod: Fast and Easy Distributed Deep Learning in TensorFlow*](https://arxiv.org/abs/1802.05799), 2018.
- Shen Li et al., [*PyTorch Distributed: Experiences on Accelerating Data Parallel Training*](https://arxiv.org/abs/2006.15704), VLDB 2020.
- Noam Shazeer et al., [*GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*](https://arxiv.org/abs/2006.16668), 2020.
- William Fedus et al., [*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*](https://arxiv.org/abs/2101.03961), JMLR 2022.
- Code: [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), [PyTorch distributed](https://github.com/pytorch/pytorch/tree/main/torch/distributed), [Horovod](https://github.com/horovod/horovod).
