# Playground

Hands-on experiments behind the blog. Mostly ML systems work: CUDA kernels,
fused / Triton kernels, single-GPU and distributed training optimization,
and data-infra ablations.

The blog posts under `../content/posts/` are the polished write-ups.
This folder is the rough notebook: source code, run scripts, raw profiles,
results CSVs and the SVGs that end up embedded in posts.

## Layout

```text
playground/
├── _template/              scaffold for new experiments (copy & rename)
├── cuda/                   raw CUDA kernels, occupancy / bandwidth studies
├── kernels/                Triton, CUTLASS, fused-op work
├── llm-training/           AMP, activation ckpt, fused optim, attention variants
├── distributed/            DDP / FSDP / TP / PP / collective-comm ablations
└── data-infra/             DataLoader, WebDataset, sharding, prefetch ablations
```

Each experiment is a self-contained bundle:

```text
<area>/<NN>-<short-slug>/
├── README.md      goal, hypothesis, run commands, result summary
├── env.txt        GPU / driver / CUDA / torch version snapshot
├── src/           python / CUDA sources
├── scripts/       run.sh / profile.sh / build.sh
└── results/       timings.csv, *.svg, optionally *.nsys-rep / *.ncu-rep
```

`NN` is a 2-digit ordinal so experiments sort lexicographically.

## Current experiments

| Area          | Experiment                    | Status     | Blog post |
|---------------|-------------------------------|------------|-----------|
| `cuda/`       | `01-vector-add`               | seed       | —         |
| `distributed/`| `01-ddp-allreduce-bench`      | seed       | —         |
| `kernels/`    | _planned_                     | —          | —         |
| `llm-training/`| _planned_                    | —          | —         |
| `data-infra/` | _planned_                     | —          | —         |

## Setup (uv)

The whole repo shares a single virtualenv, defined by the root `pyproject.toml`.
Hardware target is CUDA 13.0 + Linux (WSL2 in my case), pinned via
`[tool.uv.sources]`. Hugo / blog content is unaffected — uv only manages the
Python side.

```bash
# from repo root
uv sync                      # installs torch + numpy + plotting deps
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For CPU-only or non-Linux dev (e.g. quick edits on macOS), run
`uv sync --no-install-package torch` and install a CPU build of torch from the
default index.

## Running an experiment

Convention: every experiment has a `scripts/run.sh` that produces the artifacts
in its `results/` folder.

```bash
cd playground/cuda/01-vector-add
./scripts/run.sh
```

For profiled runs:

```bash
./scripts/profile.sh         # wraps run.sh with nsys / ncu
```

## Linking experiments to blog posts

When an experiment graduates into a blog post:

1. Copy the relevant SVGs / small CSVs into `content/posts/<slug>/` so Hugo
   bundles them with the page.
2. Add a "Reproduce" section to the post linking back to
   `https://github.com/duoan/duoan.github.io/tree/main/playground/<area>/<NN>-<slug>/`.
3. Update the **Blog post** column in the table above.

Keep large raw profiles (`*.nsys-rep`, `*.ncu-rep`) in the experiment folder
only — they are git-ignored by default, but you can opt-in per file with
`git add -f` if a particular trace is worth preserving.

## Conventions

- **Reproducibility first**: every experiment captures `env.txt` (GPU, driver,
  CUDA, torch, OS). Re-snapshot it when results land.
- **No magic numbers in posts**: numbers shown in the blog must be reproducible
  from the matching experiment's `results/` folder.
- **One experiment, one bundle**: don't grow a single bundle into a kitchen sink.
  Fork a new `NN-` directory when the question changes.
- **Notes are fine**: `notes.md` inside an experiment is a free-form scratchpad.
  The polished narrative goes into the blog post, not here.
