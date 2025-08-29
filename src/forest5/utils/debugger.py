from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from collections import Counter


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
        except Exception:  # nosec B110 - best effort close
            pass


class TraceCollector:
    """Collect trace events with optional persistence to JSONL."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = Path(directory) if directory else None
        self.events: list[dict[str, Any]] = []
        self._fh = None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._fh = (self.directory / "trace.jsonl").open("a", encoding="utf-8")

    # ------------------------------------------------------------------
    def note(
        self,
        stage: str,
        reason: str,
        at: pd.Timestamp,
        extras: dict[str, Any] | None = None,
    ) -> None:
        """Record a diagnostic note."""

        entry: dict[str, Any] = {
            "stage": stage,
            "reason": reason,
            "time": at.isoformat(),
        }
        if extras:
            entry.update(extras)
        self.events.append(entry)
        if self._fh is not None:
            json.dump(entry, self._fh, ensure_ascii=False)
            self._fh.write("\n")
            self._fh.flush()

    # ------------------------------------------------------------------
    def counts(self) -> Counter:
        """Return counts of reasons recorded so far."""

        return Counter(event.get("reason") for event in self.events)

    # ------------------------------------------------------------------
    def export_csv(self, path: Path) -> None:
        """Write all collected events to ``path`` as CSV."""

        df = pd.DataFrame(self.events)
        df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # nosec B110 - best effort close
                pass


def get_collector(debug_dir: Optional[Path]) -> TraceCollector:
    """Factory returning a :class:`TraceCollector`."""

    return TraceCollector(debug_dir)


__all__ = ["DebugLogger", "TraceCollector", "get_collector"]
