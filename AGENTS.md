# AGENTS.md

Project-specific coding guidelines live in `CLAUDE.md` (repo layout, Python/`uv`, CUDA, blog-post conventions). Read that first — it applies to all agents.

## Cursor Cloud specific instructions

This repo is a **Hugo static blog** (the deployed product) plus an **optional Python `uv` "playground"** (`playground/`) that only generates figures/data for posts and never ships.

### Services

| Service | Required? | Run / build / test |
|---|---|---|
| Hugo blog | Required | Dev server: `hugo server -D` (serves at `http://localhost:1313`, `-D` includes drafts). Prod build: `hugo --gc --minify`. |
| Python playground | Optional | Setup: `uv sync`. Run a script: `uv run python playground/<file>.py`. Lint: `uv run ruff check`. No test suite exists (pytest is a dev dep but there are no `test_*.py` files). |

### Non-obvious caveats

- **PaperMod theme is a git submodule** (`themes/PaperMod`). If `themes/PaperMod/` is empty, Hugo fails to build — run `git submodule update --init --recursive`. The startup update script handles this.
- **Hugo must be the `extended` build** (pinned `0.153.1`, matching `.github/workflows/hugo.yaml`) because PaperMod compiles SCSS via Dart Sass. Hugo (extended) and `dart-sass` are installed under `~/.local` and added to `PATH` via `~/.bashrc`; they are pinned tool binaries, so the update script does not reinstall them.
- The blog uses a theme submodule, **not** Hugo Modules, so Go is **not** required to build the site (the `go` version in the environment is irrelevant to the Hugo build).
- Playground GPU scripts (`roofline_modal.py`, `moe_perf_modal.py`) run on **Modal** and need a Modal account + network; results are already committed as JSON, so they are not needed to build the blog.
