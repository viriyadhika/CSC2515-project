#!/usr/bin/env python3
"""JSON-based metrics logger that accumulates results into a single parseable file."""
from __future__ import annotations

import json
from pathlib import Path


class MetricsLogger:
    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / "metrics.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = {}

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self._flush()

    def set_milestone(self, epoch: int, **kwargs) -> None:
        m = self._data.setdefault("milestones", {})
        m[str(epoch)] = {k: v for k, v in kwargs.items() if v is not None}
        self._flush()

    def set_final(self, **kwargs) -> None:
        self._data["final"] = {k: v for k, v in kwargs.items() if v is not None}
        self._flush()

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2))
