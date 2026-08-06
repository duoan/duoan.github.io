"""ARGUS: production-scale tracing & progressive diagnosis (paper reimplementation).

Architecture mirrors Zhou et al., arXiv:2606.20374:

* ``producer``  — Trace Producer (§4): semantics, CUPTI kernels, CPU stacks
* ``processor`` — Processor (§5): KDE compression, Perfetto export, ingestion
* ``storage``   — tiered metrics + object store (§5)
* ``analysis``  — progressive diagnosis L1–L3 (§6)
* ``client``    — FT-Client-lite CLI

See ``argus/README.md`` and ``argus/modal_stack.py`` for Docker / Modal usage.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
