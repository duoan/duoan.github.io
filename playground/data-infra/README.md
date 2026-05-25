# Data-infra experiments

`DataLoader`, WebDataset, sharding, prefetch, and storage-layer ablations.
The question this folder asks over and over: *is the GPU actually getting
fed?*

## Index

_(no experiments yet — this directory is a placeholder.)_

## Planned

| Slug                       | Question                                                    |
|----------------------------|-------------------------------------------------------------|
| `01-num-workers-sweep`     | Sweep `num_workers` and `prefetch_factor`, measure step-time jitter. |
| `02-pinned-mem-vs-not`     | Pinned memory + `non_blocking=True` impact on H2D overlap.   |
| `03-webdataset-shard-size` | Throughput vs. shard size for a WebDataset stream.            |
| `04-image-decode-cpu-vs-gpu` | CPU PIL vs. NVJPEG decode for a vision pipeline.             |

## Patterns

- Always report **step-time jitter** (e.g. p99/p50 ratio), not just mean.
  A pipeline that's "fast on average" but bursts can starve the GPU more
  than a slower-but-steady one.
- Profile with `torch.profiler` first to confirm `DataLoader` is actually
  the bottleneck before you tune knobs.
- Compare against a `null` data source (synthetic tensors with the same
  shape on-device) to find the upper bound on tokens/s for the model.
  If the real pipeline can't get within ~10% of that, it's a data problem.
