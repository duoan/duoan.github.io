---
title: "Megatron Internals III: Mixed Precision, Loss Scaling, and Grad Clipping"
date: 2025-04-26
tags: ["LLM", "Megatron", "Mixed Precision", "AMP", "Loss Scaling", "Optimizer"]
categories: ["Engineering"]
draft: false
series: ["LLM Training"]
cover:
  image: amp_flow.svg
  alt: "Mixed precision training flow with fp32 master weights and loss scaling"
  relative: true
---

# Megatron Internals III: Mixed Precision, Loss Scaling, and Grad Clipping

Mixed precision training is sometimes described as "use fp16 for speed."
That is incomplete.
The working system is a contract between three copies of state: low-precision tensors for throughput, fp32 master tensors for stable updates, and distributed checks that prevent one bad rank from corrupting the step.
Megatron's `Float16Optimizer` pattern is a good way to understand that contract.

## TL;DR

- Mixed precision saves memory and uses Tensor Cores, but the optimizer still updates fp32 master parameters.
- Static memory includes model parameters, gradients, master parameters, and optimizer moments; activation memory is separate and often larger for long sequences.
- Loss scaling shifts small fp16 gradients into a representable range; dynamic loss scaling raises or lowers the scale based on overflow checks.
- Overflow detection must be synchronized across data and model-parallel ranks before any rank updates.
- Gradient clipping happens after unscale and fp32 gradient copy, and the norm must include all model-parallel shards.
- This post follows [Part I](../megatron-distributed-init/) and [Part II](../megatron-model-parallel-internals/).

## 1. The memory inventory

A training step stores more than weights.
For each parameter, Adam-style training may hold:

- model parameters used by forward and backward;
- gradients produced by backward;
- fp32 master parameters used by the optimizer;
- first and second moment estimates;
- activation tensors needed for backward;
- temporary buffers for communication and fused kernels.

The first four are model states.
Activations and temporary buffers are residual states.
ZeRO's original framing is still useful: optimizer states, gradients, and parameters are the static part; activations and buffers depend on batch shape, sequence length, recomputation, and parallel schedule.

![Mixed precision keeps fast tensors and stable tensors](precision_memory_table.svg)

For a simple fp16 Adam setup, per parameter static storage can look like:

```text
fp16 model param     2 bytes
fp16 grad            2 bytes
fp32 master param    4 bytes
fp32 Adam m          4 bytes
fp32 Adam v          4 bytes
----------------------------
total               16 bytes
```

This is before activation memory.
For long-context training, activations can dominate the same way MoE activations dominate production sparse runs; see the modern MoE performance post for a recent example of activation pressure in large systems: [Large MoE Performance](../large-moe-from-sparsity-to-communication/).

## 2. Why keep fp32 master weights?

Low precision is fast because modern GPUs have high-throughput Tensor Cores for fp16, bf16, and newer formats.
But optimizer updates are accumulation-heavy.
Small updates can disappear if they are applied directly to fp16 parameters.
The classic mixed precision recipe therefore keeps two parameter copies:

```text
model_param: fp16 or bf16, participates in forward/backward
main_param:  fp32, receives optimizer update
```

At the start of an iteration, the model copy reflects the master copy.
The forward and backward pass run mostly in model precision.
Before the optimizer step, gradients are copied or accumulated into fp32 buffers, unscaled, checked, clipped, and applied to the master weights.
The low-precision model weights are refreshed from the master weights.

![AMP training flow with fp32 master weights](amp_flow.svg)

This explains an otherwise surprising point: mixed precision reduces activation and compute precision, but it does not eliminate fp32 state.
It moves fp32 to the places where it pays for stability.

## 3. fp16 versus bf16

fp16 and bf16 are both 16-bit formats.
They fail differently.
fp16 has more mantissa precision but a much narrower exponent range.
bf16 keeps the fp32 exponent range and sacrifices mantissa bits.
For training, exponent range is often more valuable than extra mantissa precision.
That is why bf16 usually trains without loss scaling, while fp16 commonly needs it.
The rest of this post uses fp16 as the harder case.
The architecture still applies to bf16, but the dynamic scaling path is often disabled or less important.

## 4. Loss scaling in one equation

Backprop computes gradients proportional to the loss.
If gradients are too small for fp16, they underflow to zero.
Loss scaling multiplies the loss by a factor `S` before backward:

```text
scaled_loss = S * loss
scaled_grad = S * grad
grad = scaled_grad / S
```

The math is unchanged if we divide by `S` before the optimizer update.
The representation is changed during backward, where fp16 needs help.
Static loss scaling chooses one fixed `S`.
Dynamic loss scaling adapts `S` during training.

## 5. Dynamic loss scaling as feedback control

Dynamic scaling has only two signals:

- gradients were finite;
- at least one gradient contained `inf` or `nan`.

If all gradients are finite for many consecutive steps, increase the scale.
If any gradient overflows, skip the update and decrease the scale.

![Dynamic loss scaling is a feedback controller](dynamic_loss_scale.svg)

A minimal controller:

```python
if found_inf:
    loss_scale = max(loss_scale / backoff_factor, min_loss_scale)
    growth_tracker = 0
    skip_optimizer_step = True
else:
    growth_tracker += 1
    if growth_tracker == growth_interval:
        loss_scale *= growth_factor
        growth_tracker = 0
```

The skipped step matters.
Once an overflow happened, the gradients are not trustworthy.
Applying them with a smaller scale after the fact would not recover the true values.

## 6. Distributed overflow is a global decision

In single-GPU AMP, overflow detection is local.
In Megatron, no rank can update alone.
Consider a tensor-parallel row-parallel layer.
One rank may see a finite partial gradient while another rank overflows.
If the finite rank updates its shard and the overflowed rank skips, the distributed parameter no longer represents one coherent model.
Therefore overflow flags must be reduced across the relevant parallel groups.
Conceptually:

```python
found_inf_local = check_grads_for_inf_or_nan(local_grads)
found_inf = all_reduce_max(found_inf_local, model_and_data_parallel_groups)

if found_inf:
    skip_step_on_every_rank()
```

Different implementations choose different exact groups, but the invariant is simple: all ranks participating in one optimizer step must agree.

## 7. The `Float16Optimizer` pattern

Megatron-style mixed precision wraps a base optimizer.
The wrapper owns the state transitions around it.
At a high level:

```text
Float16Optimizer
  - keeps references to fp16/bf16 model params
  - builds fp32 main params for optimizer updates
  - copies model grads to main grads
  - unscales gradients
  - checks overflow
  - clips gradients
  - calls inner optimizer.step()
  - copies main params back to model params
```

The inner optimizer can remain a familiar Adam variant.
The wrapper is what makes the dtype and distributed invariants explicit.
This separation is useful because tensor-parallel layers from [Part II](../megatron-model-parallel-internals/) only own shards.
The optimizer wrapper does not need to understand the math of each layer.
It only needs parameter lists grouped by dtype and parallel ownership.

## 8. Copying gradients is not just casting

Backward produces gradients attached to model-precision parameters.
Before the optimizer step, those gradients are moved to fp32 master parameters.
Three things can happen at once:

1. cast fp16 gradient to fp32;
2. divide by the current loss scale;
3. place the result in the master gradient buffer.

That buffer may be flattened for efficiency.
It may also be sharded by ZeRO or distributed optimizer logic.
The visible concept remains:

```python
main_grad.copy_(model_grad.float())
main_grad.mul_(1.0 / loss_scale)
```

After this point, gradient clipping and optimizer math should see unscaled fp32 gradients.

## 9. Gradient clipping under model parallelism

Clipping by global norm computes:

```text
global_norm = sqrt(sum_i ||grad_i||^2)
clip_coef   = max_norm / (global_norm + eps)
```

In a tensor-parallel model, each rank owns only some parameters.
The local norm is incomplete.
Megatron computes local squared norms, then reduces the sum across the model-parallel and data-parallel topology required by the optimizer.
Only after the global norm is known can each rank scale its local gradients.

![Gradient clipping under model parallelism](grad_clip_with_mp.svg)

The order is important:

1. detect overflow on scaled gradients;
2. copy to fp32 main gradients;
3. unscale by loss scale;
4. compute distributed norm;
5. clip local fp32 gradients with global coefficient;
6. step fp32 master parameters;
7. copy master parameters back to model precision.

Clipping before unscale would make the threshold depend on the current loss scale.
Clipping only local shards would make the threshold depend on TP size.
Both are wrong.

## 10. What happens with ZeRO

ZeRO changes where optimizer state lives.
It does not remove the need for the dtype choreography.
With ZeRO-1, optimizer states are partitioned across DP ranks.
With ZeRO-2, gradients are partitioned too.
With ZeRO-3, parameters are partitioned.
The mixed-precision wrapper still needs to know:

- where the fp32 master shard lives;
- where the model-precision shard lives during forward;
- where unscaled gradients should be reduced or partitioned;
- which ranks must agree on overflow and clipping.

This is why initialization topology from [Part I](../megatron-distributed-init/) is a prerequisite for optimizer logic.
The optimizer's memory savings are defined over DP-style redundancy, while TP and PP define model ownership.

## 11. Activation memory is a separate battle

Mixed precision reduces activation bytes when activations are stored in fp16 or bf16.
But activation memory can still dominate.
The common tools are:

- activation checkpointing, which recomputes forward pieces during backward;
- selective recomputation, which targets cheap tensors;
- sequence parallelism, which shards activations across TP ranks;
- pipeline scheduling, which limits how many microbatch activations are resident.

Loss scaling and fp32 master weights do not solve activation pressure.
They solve numerical stability and static-state update precision.
Conflating these two memory problems leads to bad debugging.
If a job OOMs before backward, loss scaling is not the lever.
If gradients turn into zeros or `inf`, activation checkpointing is not the lever.

The design is simple when stated as an invariant: fast tensors compute; stable tensors update; all ranks agree before stepping.

## References

- Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740), ICLR 2018.
- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), 2019.
- Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), 2021.
- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), SC 2020.
- IEEE, [754-2019 Standard for Floating-Point Arithmetic](https://ieeexplore.ieee.org/document/8766229), 2019.
