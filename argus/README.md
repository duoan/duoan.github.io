# ARGUS full-stack framework

Open reimplementation of the architecture in
[ARGUS: Production-Scale Tracing and Performance Diagnosis for over 10,000-GPU Clusters](https://arxiv.org/abs/2606.20374)
(Zhou et al., Tencent, arXiv 2606.20374).

This is a **runnable research stack**, not the production Tencent deployment.
It mirrors the paper’s four planes so you can exercise the data path end-to-end
on a laptop, in Docker, or on Modal GPUs.

```
Trace Producer (§4)  →  Processor (§5)  →  Tiered Storage (§5)  →  Diagnosis (§6)
  CPU stacks              KDE compress         metrics JSONL           L1 iteration
  semantics (CUDA Ev)     Perfetto export      object store            L2 phases
  CUPTI kernels           Unix-socket ingest                           L3 W₁ kernels
```

## Layout

| Path | Role |
|---|---|
| `producer/` | Trace Producer: semantics, CUPTI / fake kernels, stack sampler |
| `processor/` | KDE compression, Perfetto JSON, Unix-socket pipeline |
| `storage/` | File-backed metrics + object store |
| `analysis/` | Progressive diagnosis L1 / L2 / L3 |
| `client/` | FT-Client-lite CLI |
| `native/cupti_tracer.cpp` | Minimal CUPTI Activity API shared library |
| `docker/` | CPU + CUDA Dockerfiles, compose, entrypoint |
| `sim.py` | Multi-rank synthetic straggler job |
| `modal_stack.py` | Modal e2e (Dockerfile image + optional CUPTI) |

Blog-oriented remasurements stay under `playground/argus_*` (figures for the
paper-reading post). This package is the reusable framework those demos point at.

## Quick start (local, no GPU)

```bash
# from repo root
uv sync
uv run pytest argus/tests -q
uv run python -c "from argus.sim import run_synthetic_job; \
  import tempfile,json; d=tempfile.mkdtemp(); \
  print(json.dumps({k:run_synthetic_job(data_dir=d)[k] for k in \
  ('detected','flagged_ranks','mean_compression_ratio','notes')}, indent=2))"
```

## Docker

```bash
# one-shot e2e synthetic job
docker build -f argus/docker/Dockerfile -t argus:cpu .
docker run --rm -v "$PWD/argus-data:/data" -e ARGUS_JOB_ID=synth argus:cpu sim

# compose: processor + two producers + analyzer
docker compose -f argus/docker/docker-compose.yml up --build

# CUDA / CUPTI image (needs nvidia-container-toolkit)
docker build -f argus/docker/Dockerfile.cuda -t argus:cuda .
```

Entrypoint roles: `sim` | `processor` | `producer-demo` | `analyze` | `shell`.

## Modal

```bash
uv run python argus/modal_stack.py             # local synthetic (no Modal token)
uv run modal run argus/modal_stack.py           # CPU stack via Dockerfile image
uv run modal run argus/modal_stack.py --cupti   # real CUPTI on Modal GPU
```

Writes `argus/modal_results.json`. `modal run` needs a Modal token; without one,
use the plain Python entry (same synthetic job the Modal CPU path runs).

## FT-Client-lite

```bash
uv run python -m argus.client.cli --data-dir /data/metrics --job-id synth
```

## What is intentionally stubbed

| Paper component | Here |
|---|---|
| py-spy streaming stacks | `FakeStackSampler` + optional `PySpyStackSampler` |
| Vector log shipper | Direct in-process / Unix-socket ingest |
| Prometheus + Grafana | JSONL metric store (same schema, no TSDB) |
| Go Processor (7.3k LoC) | Python Processor with the same responsibilities |
| `CUDA_INJECTION64_PATH` injector | Explicit `CuptiTracer` start/stop API |
| FT-Client UI | CLI that prints L1–L3 + writes `alerts.jsonl` |

The algorithms that matter for the paper claims — KDE valley clustering,
log-normal mixture CDF reconstruction, W₁ + IQR L3, L1 jitter/change-point,
L2 CV/z-score — are implemented and covered by `argus/tests`.
