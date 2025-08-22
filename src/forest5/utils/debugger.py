from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DebugLogger:
    """Simple JSONL logger for debugging trading decisions."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "decision_log.jsonl"
        self._fh = self.file.open("a", encoding="utf-8")

    def log(self, event: str, **data: Any) -> None:
        entry = {"event": event, **data}
        json.dump(entry, self._fh, ensure_ascii=False)
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
