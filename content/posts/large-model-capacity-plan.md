---

title: "From Scaling Laws to Cluster Size: A Practical Guide to Planning Large-Scale Model Training"
date: 2025-02-02
tags: ["LLM", "Distributed Training", "Scaling Laws", "Capacity Planning", "Systems"]
categories: ["Engineering", "Research"]
draft: false
---

## Why Capacity Planning Is the Hardest Part of Large Model Training

Before you write a single line of training code, you must answer a few brutal questions:

- How many tokens do I actually need?
- What sequence length should I train on?
- How many GPUs will this take?
- How long will it run?
- What parallelism strategy makes this feasible?

Most teams get this wrong — not because they lack theory,
but because they never connect **scaling laws → systems constraints**.

This post walks through a **practical, engineer-first workflow** to estimate
distributed training requirements for large language and multimodal models.

Not exact.  
But accurate enough to avoid catastrophic planning mistakes.

---

## Step 1: Decide What You’re Scaling *For*

Scaling laws don’t tell you *what to train* — only how loss behaves as you scale.

So start by fixing **intent**, not hardware:

- Are you training a **base model** or an **instruction-tuned model**?
- Is reasoning depth important, or is this a retrieval-heavy model?
- Is context length a core capability, or a nice-to-have?

These choices determine:

- token count,
- sequence length,
- architecture,
- and parallelism constraints later.

---

## Step 2: Estimate Token Budget from Scaling Laws (Order of Magnitude)

Empirically (Kaplan-style, later refined by Chinchilla), optimal training roughly follows:

```text
tokens ≈ 20 × model_parameters
````

This is not a law of physics — it’s a **planning heuristic**.

### Example

| Model Size | Params | Token Budget |
| ---------- | ------ | ------------ |
| 1B         | 1e9    | ~20B tokens  |
| 7B         | 7e9    | ~140B tokens |
| 30B        | 3e10   | ~600B tokens |

If you’re compute-constrained, you may undershoot.
If you’re data-rich, you may overshoot slightly.

But if you’re off by **10×**, your plan is wrong.

---

## Step 3: Choose Sequence Length (This Dominates Everything)

Sequence length is the most underestimated variable in training planning.

Key facts:

- Attention FLOPs scale **quadratically** with sequence length.
- Memory scales roughly linearly (but with large constants).
- Longer context reduces effective batch size.

You should choose sequence length based on **actual usage**, not benchmarks.

Typical regimes:

| Use Case                  | Sequence Length |
| ------------------------- | --------------- |
| Classic LLM               | 2K–4K           |
| Instruction / Reasoning   | 4K–8K           |
| Long-context / Multimodal | 8K–32K          |

Once chosen, **everything downstream is constrained by this decision**.

---

## Step 4: Convert Tokens → Training Steps

Given:

- total tokens `T`
- sequence length `L`
- global batch size `B` (in sequences)

```text
steps = T / (B × L)
```

Example:

- T = 300B tokens
- L = 4K
- B = 2048 sequences

```text
steps ≈ 300e9 / (2048 × 4096) ≈ 35,800 steps
```

This number should immediately feel plausible or alarming.
If it doesn’t, recheck your assumptions.

---

## Step 5: Estimate FLOPs (Reality Check)

A rough rule of thumb for decoder-only transformers:

```text
FLOPs ≈ 6 × params × tokens
```

So for a 30B model, 600B tokens:

```text
FLOPs ≈ 6 × 3e10 × 6e11 = 1.08e23 FLOPs
```

Now compare to hardware:

- A100 (80GB): ~312 TFLOPs (bf16 peak)
- Effective utilization: 30–50% in real training

This immediately tells you whether your plan is *weeks*, *months*, or *never*.

---

## Step 6: Translate FLOPs → GPU-Hours

Let’s assume:

- 40% sustained utilization
- 125 TFLOPs effective per GPU

```text
GPU-seconds ≈ total_FLOPs / effective_TFLOPs
GPU-hours ≈ GPU-seconds / 3600
```

This gives you a **budget-level estimate**.
If this number scares you, good — that’s the point.

---

## Step 7: Choose Parallelism Based on Constraints (Not Preference)

Parallelism is not aesthetic.
It is forced by **memory, sequence length, and cluster topology**.

### Data Parallel (DP)

- Best scaling efficiency
- Limited by model + optimizer state size
- Breaks first with large models

### Tensor Parallel (TP)

- Splits large matrices
- Required once model doesn’t fit in a single GPU
- Communication-heavy but unavoidable

### Pipeline Parallel (PP)

- Useful when TP alone isn’t enough
- Increases bubble overhead
- Complicates scheduling and checkpointing

### Sequence Parallel / Context Parallel

- Needed for long-context models
- Reduces activation memory
- Adds collective ops inside attention

### Rule of Thumb

1. **Fit model** → TP
2. **Fit activations** → SP / CP
3. **Scale throughput** → DP
4. **Only then consider PP**

If you start with pipeline parallelism first, you probably misplanned earlier.

---

## Step 8: Back Into Cluster Size

Once you know:

- per-step time,
- steps required,
- target wall-clock time,

you can estimate cluster size:

```text
GPUs ≈ total_GPU_hours / target_hours
```

Then ask:

- Can my network handle the all-reduces?
- Can my storage feed this many workers?
- Can I checkpoint at this scale?

If the answer is “no”, **reduce ambition, not code quality**.

---

## Step 9: Embrace Staged Training (Reality Strategy)

Almost no successful large model is trained in one monolithic run.

Common strategies:

- shorter sequence first, then extend,
- freeze parts of the model early,
- train projector / adapters separately,
- mix offline distillation stages.

Scaling laws guide *direction*, not execution.

---

## A Concrete Example: Planning a 30B Scale-Up VLM Run

Let’s make this tangible — and painful.

Assume we are training a **30B Parameter VLM** (roughly Llama-3-30B class + Vision). This crosses the critical threshold where a model **no longer fits on a single GPU**, forcing us to deal with real distributed system constraints.

### Target Model

- **Vision encoder**: SigLIP-SO400M / InternViT-6B (Frozen or LoRA)
- **Language model**: 30B parameters (Decoder-only)
- **Precision**: bfloat16
- **Total trainable params**: ~30B

### Step 1: Token Budget (The "Chinchilla" Bill)

Using the standard optimal compute heuristic ():

```text
tokens ≈ 20 × 30e9 ≈ 600B tokens

```

*Note: For a production foundation model, you typically want to "over-train" (e.g., Llama 3 style) beyond Chinchilla optimal, often hitting 1T+ tokens. But let's stick to **600B** for a resource-constrained optimal plan.*

---

### Step 2 & 3: Sequence Length & Context Strategy

For a 30B model, training on short context (2K) is a waste of its reasoning potential. We need at least **4K** or **8K** to handle multi-image reasoning or document understanding.

Let’s fix:

- **Sequence Length**: 4096 (4K)
- **Data Mix**: Interleaved images + text.

---

### Step 4: Training Steps

We need a large Global Batch Size (GBS) to maintain training stability for a 30B model. A typical GBS is ~2M to 4M tokens.

Let’s aim for **4M tokens per step**.

- Batch size in sequences:  sequences.
- Let’s round to **Global Batch Size = 1024**.

Total steps required:

```text
Total Steps = Total Tokens / Tokens_per_step
Steps = 600e9 / (1024 × 4096)
Steps ≈ 143,000 steps

```

This is a **long** training run.

---

### Step 5: FLOPs Estimate (The Reality Check)

```text
FLOPs ≈ 6 × params × tokens
FLOPs ≈ 6 × 30e9 × 600e9
FLOPs ≈ 1.08e23 (108 ZettaFLOPs)

```

To put this in perspective: This is roughly **100×** the compute of the 3B example.

---

### Step 6: GPU-Time Estimate

Assume **NVIDIA A100 (80GB)**.

- Peak BF16: 312 TFLOPs.
- **Effective TFLOPs**: Let's be realistic. With 30B params, communication overhead (All-Gather/Reduce) increases. Let's assume **130 TFLOPs** sustained (approx 42% MFU).

```text
GPU-seconds ≈ 1.08e23 / 1.3e14 ≈ 8.3e8 seconds
GPU-hours ≈ 230,000 GPU-hours

```

This number is the most important output of the planning phase.
**230,000 GPU-hours.**

If you rent AWS p4d instances (~$4/hour/GPU), this run costs roughly **$1 Million USD**.

---

### Step 7: Back Into Cluster Size

We cannot run this on 8 GPUs. It would take 3+ years.

Target Training Time: **3 Weeks (approx 500 hours)**.

```text
Required GPUs = Total GPU-hours / Target Hours
GPUs = 230,000 / 500 = 460 GPUs

```

Rounding to nearest convenient cluster size (multiples of node size, e.g., 8 GPUs/node):
**Target: 64 Nodes (512 GPUs).**

With 512 GPUs, training takes:
.

*Verdict: Manageable, but requires a robust checkpointing strategy.*

---

### Step 8: Parallelism Strategy (The Critical Engineering)

Here is where the 30B model differs from the 3B model.

**Memory Constraints:**

- **Model Weights (bf16)**: 30B x 2 bytes = 60 GB
- **Optimizer State (AdamW, fp32)**: 30B x 8 bytes = 240 GB
- **Gradients (bf16)**: 30B x 2 bytes = 60 GB
- **Activations**: Varies by batch size and seq len.

**Total Static Memory required per copy**: ~360 GB.
**Available Memory per GPU**: 80 GB.

**Conclusion:** The model does NOT fit on one GPU. We *must* shard.

#### The Configuration: 3D Parallelism

We have **512 GPUs** (64 Nodes  8 GPUs). We need to fit the model and maximize throughput.

**1. Tensor Parallelism (TP):**
We need to shard weights to fit memory and reduce latency.

- Set **TP = 4**.
- This splits the 30B model across 4 GPUs.
- Each GPU holds ~1/4 of weights and optimizer states.
- *Why not TP=8?* TP requires high-bandwidth NVLink. TP=4 allows us to fit 2 model replicas per node (8 GPUs), reducing inter-node communication.

**2. Pipeline Parallelism (PP):**

- Set **PP = 1** (None).
- With TP=4, the model fits comfortably in memory. PP introduces "bubbles" (idle time). Avoid it if possible.

**3. Data Parallelism (DP):**

- Total GPUs = 512.
- GPUs per Model Replica = TP x PP = 4 x 1 = 4.
- Total Replicas (DP size) = 512 / 4 = 128.

**Final Config:**

- **TP = 4** (Intra-node)
- **DP = 128** (Inter-node, potentially using ZeRO-1 to shard optimizer states further if activation memory gets tight)
- **Global Batch Size**: 1024
- **Micro Batch per Replica**: 1024 / 128 = 8.

This configuration ensures:

1. Model fits in HBM.
2. All tensor splitting happens over fast NVLink (inside the node).
3. Gradient synchronization happens across the network (Infiniband/EFA).

---

### Step 9: Storage & Checkpointing (The Hidden Killer)

With **512 GPUs** writing data simultaneously:

- Checkpoint size: ~360 GB (BF16 weights + FP32 optimizer).
- If 128 DP ranks try to write effectively the same data (or sharded ZeRO states) to a shared NFS/S3: **You will crash the storage.**

**Mitigation Plan:**

- Use **Async Checkpointing** (CPU offload upload).
- Save only rank 0 for weights (consolidated) or use distcp tools.
- Frequency: Every 500 steps (approx every 2 hours).
- Best practice on AWS: <https://github.com/awslabs/s3-connector-for-pytorch>

---

### What This Example Should Teach You

1. Scaling laws give **order-of-magnitude**, not exact answers.
2. Sequence length matters more than people think.
3. FLOPs math prevents fantasy planning.
4. Parallelism is *forced* by memory and context, not preference.
5. Most VLMs are **capacity-planning problems**, not modeling problems.

If you can’t estimate this on paper,
you’re not ready to launch the training job.

---

## Closing: Scaling Is a Systems Problem Disguised as Math

Scaling laws tell you **what is theoretically efficient**.
Distributed systems tell you **what is physically possible**.

Good training plans live at the intersection.

If you can:

- estimate tokens,
- estimate FLOPs,
- estimate memory,
- and choose parallelism intentionally,

you’re already ahead of most teams trying to “just train and see”.

The rest is engineering.

---

*In the next post, I’ll walk through concrete parallelism configurations
(TP × DP × SP × PP) for real cluster shapes, and how small planning mistakes
explode into 10× cost overruns.*

- *“Why Long Context Forces Sequence Parallelism (and When It’s Not Worth It)”*
- *“Why Pipeline Parallelism Is a Last Resort”*
- *“Capacity Planning Mistakes I’ve Seen in Real Training Runs”*
