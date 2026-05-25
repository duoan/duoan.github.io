# playground

Where I write code. Mostly ML systems stuff — CUDA, kernels, training,
data — but no fixed layout. Dump files in here however makes sense at
the time; reorganize when it actually starts to hurt.

Hugo doesn't look at this folder, so nothing here ships to the blog.

## Setup

`pyproject.toml` lives at the repo root. From the repo root:

```bash
uv sync
uv run python whatever.py
```

torch is pinned to the cu130 wheels on Linux/Windows; macOS gets a CPU
build (fine for editing, not for running).

For C/C++/CUDA, just `nvcc whatever.cu -o whatever && ./whatever`.

## Notes

- `__pycache__/`, `.venv/`, `*.nsys-rep`, `*.ncu-rep`, `build/` etc. are
  already in the repo `.gitignore`.
- When something here turns into a blog post, copy the relevant figures
  into `content/posts/<slug>/` and link back to this folder from the post.
