# `<NN>-<short-slug>`

> Copy this whole `_template/` folder, rename to `<NN>-<short-slug>` under the
> right area, then fill in the sections below.

## Question

One sentence: what specifically is this experiment trying to find out?

## Hypothesis

What I expect to see, and roughly why. State this *before* running.

## Setup

- GPU / driver / CUDA / torch versions are captured in `env.txt`
  (regenerate with `./scripts/snapshot_env.sh`).
- Anything not captured by `env.txt` (dataset, model size, etc.) goes here.

## Run

```bash
./scripts/snapshot_env.sh > env.txt
./scripts/run.sh
```

Outputs land in `results/`.

## Results

Brief summary, headline numbers, and one or two key plots. Detailed
narrative belongs in the blog post, not here.

## Follow-ups

Open questions and what the next experiment should ask.
