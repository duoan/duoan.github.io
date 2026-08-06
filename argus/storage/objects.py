"""Filesystem object store for Perfetto traces (§5 tiered storage)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ObjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_json(self, key: str, payload: dict[str, Any]) -> Path:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    def put_bytes(self, key: str, data: bytes) -> Path:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def get_json(self, key: str) -> dict[str, Any]:
        return json.loads((self.root / key).read_text(encoding="utf-8"))

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self.root / prefix if prefix else self.root
        if not base.exists():
            return []
        out = []
        for p in base.rglob("*"):
            if p.is_file():
                out.append(str(p.relative_to(self.root)))
        return sorted(out)
