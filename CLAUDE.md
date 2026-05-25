# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Project-Specific Guidelines

This repo = a Hugo blog (`content/posts/`) plus a free-form `playground/` for ML systems code. The principles above still apply; these are the project-specific facts that catch out new edits.

### Layout

- **`playground/` stays flat.** No framework, no `_template/`, no per-experiment scaffolding directories. Add files where they make sense at the time; reorganize only when it actually starts to hurt.
- **No code under `content/`.** Hugo only renders `content/`. Source code lives in `playground/`. Figures a post embeds belong inside that post's page bundle (`content/posts/<slug>/`).
- **Don't modify `themes/PaperMod`** — it's an upstream git submodule. Theme overrides go in `layouts/` at the repo root.

### Python

- Deps live in the root `pyproject.toml`, managed by `uv`. Don't introduce `pip install`, `requirements.txt`, `poetry`, or `conda`.
- `torch` is pinned to the `cu130` index via `[tool.uv.sources]` for Linux/Windows; macOS pulls CPU torch from PyPI.
- Format and lint with `ruff` only (already wired into the editor). Don't add `black`, `isort`, or `flake8`.

### CUDA / C++

- Prefer the simplest build path:
  - PyTorch-side: `torch.utils.cpp_extension.load_inline` in a single `.py`.
  - Standalone: a single `.cu` + `nvcc foo.cu -o foo`.
- Reach for `Makefile` / CMake only when custom flags or multiple translation units actually require it.
- Format with `clang-format` (config at repo root).

### Blog posts

- New post = page bundle: `content/posts/<slug>/index.md` plus sibling figures (`*.svg`, `*.csv`). Reference figures by relative filename.
- Frontmatter must include `title`, `date`, `tags`, `categories`, `draft`. Match the style of existing posts.
- **Historical posts are immutable.** Don't update `cu128` → `cu130` or other version strings on old posts; they record the actual environment at writing time.

### What goes where in git

- **Don't commit**: Hugo build output (`public/`, `resources/`, `.hugo_build.lock`); Python caches (`.venv/`, `__pycache__/`, `.ruff_cache/`, `.pytest_cache/`).
- **Commit via Git LFS** (see `.gitattributes`): profiler outputs (`*.nsys-rep`, `*.ncu-rep`, `*.qdrep*`, `*.qdstrm`, `*.sqlite`). These are experiment data, not throw-away artifacts — keep them. Don't move them into `.gitignore`.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
