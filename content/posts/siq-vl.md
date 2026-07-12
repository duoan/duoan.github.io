---
title: "SiQ-VL: Training a Reasoning-Capable VLM When You’re GPU-Poor"
date: 2025-12-05
tags: ["VLM", "Multimodal", "Reasoning", "Distillation", "SigLIP", "Qwen"]
categories: ["Engineering", "Research"]
draft: true
---

> **Superseded.** This note is folded into [SiQ-VL: A Curriculum for Small VLMs When Compute Is the Hard Constraint](/posts/siq-vl-curriculum-under-compute-constraints/). Kept as draft for git history only.

## When You Can’t Afford Scale

Most modern Vision-Language Models (VLMs) are built under one assumption:  
**you have access to massive GPU clusters**.

But what if you don’t?

**SiQ-VL** started from a very practical question:

> *How far can we push a small Vision-Language Model when compute is the hard constraint?*

Instead of scaling parameters or training end-to-end, we focused on:

- freezing aggressively,
- training in stages,
- and injecting reasoning via **offline Chain-of-Thought (CoT) distillation**.

The result is a lightweight VLM that demonstrates **emergent reasoning behavior** under strict compute limits.

Repo: <https://github.com/duoan/SiQ_VL>

---

## The Smallest VLM We Could Afford

SiQ-VL follows a deliberately simple architecture:

- **Vision encoder**: SigLIP-2 (frozen)
- **Language model**: Qwen2.5 (0.5B / 1.5B)
- **Connector**: a lightweight linear projector with pixel shuffle

No perceivers.  
No giant MLPs.  
No end-to-end finetuning.

This “connect-the-dots” design is intentional: every moving part is easy to reason about, profile, and debug.

---

## Why Vision Tokens Were the First Problem

Vision encoders are expensive—not because of parameters, but because of **token count**.

A 384×384 image with a ViT-style encoder produces **729 visual tokens**.  
Feeding all of them directly into an LLM is quadratic pain.

So we compressed early.

### Pixel Shuffle Projection

Instead of pooling or attention-based resampling, we use a **pixel shuffle projector**:

- merge every 2×2 patch group,
- reduce token count by **4×**,
- preserve spatial locality,
- then apply a linear projection.

Typical shape transition:

- `[B, 729, D] → [B, 182, 4D] → [B, 182, hidden_size]`

This single decision gave us:

- faster training,
- lower memory usage,
- and more stable optimization.

---

## Train in Stages or Don’t Train at All

Trying to learn everything at once is the fastest way to fail on limited hardware.

SiQ-VL uses a **three-stage training pipeline**, each stage solving one problem at a time.

---

### Stage 1: Projector Alignment

**Goal:** teach the projector to translate vision features into the LLM’s embedding space.

- Vision: frozen  
- LLM: frozen  
- Trainable params: projector only  

This stage converges fast—but outputs are often gibberish.  
That’s expected: alignment ≠ instruction following.

---

### Stage 2: Multimodal Instruction Tuning

Now we let the language model adapt.

- Vision: still frozen  
- LLM: LoRA-tuned  
- Projector: continues training  

This stage fixes:

- mixed-language artifacts,
- repetition loops,
- instruction failures.

Only after Stage 2 does the model become a usable VLM.

---

### Stage 3: Reasoning via Offline CoT Distillation

Small models don’t reason well zero-shot.  
Running large teachers online is expensive.

So we distill **offline**.

We generate Chain-of-Thought traces from multiple teacher models:

- Qwen3-VL-Thinking
- InternVL
- HunyuanOCR

Each teacher contributes a different reasoning bias:

- structured math,
- chart understanding,
- OCR-heavy visual logic.

The student is trained on `(image, question, rationale, answer)` tuples—without ever running the teacher during training.

This is the key to making reasoning affordable.

---

## Why Offline Distillation Matters

Offline CoT distillation gives us three advantages:

1. **Memory efficiency**  
   No teacher model in GPU memory during training.

2. **Curriculum control**  
   We choose when reasoning appears in the pipeline.

3. **Multi-teacher diversity**  
   Different teachers inject different inductive biases.

Under compute constraints, this mattered more than model size.

---

## What Actually Improved (and What Didn’t)

A few concrete observations from experiments:

- LoRA tuning only **4–5%** of parameters was enough
- Reasoning traces became **longer and more structured**
- Hallucination rate dropped
- Running accuracy became more stable

Interestingly:

- ROUGE-L barely changed  
- but qualitative reasoning clearly improved

This reinforced a lesson we keep relearning:
> **surface metrics don’t fully capture reasoning quality**

---

## What This Project Is *Not*

SiQ-VL is not:

- state-of-the-art,
- fully benchmarked,
- or production-ready.

It *is* a proof that:

- reasoning signals can be distilled,
- small VLMs can benefit disproportionately from curriculum design,
- and data quality beats brute-force scaling when GPUs are scarce.

---

## Takeaways

If you’re GPU-poor, your leverage points are:

- freeze aggressively,
- reduce token count early,
- separate alignment from reasoning,
- and distill offline whenever possible.

SiQ-VL is less about architecture novelty, and more about **engineering discipline under constraints**.

Sometimes, that’s enough to make something interesting happen.

Repo: <https://github.com/duoan/SiQ_VL>
