from __future__ import annotations

import threading
import time
from pathlib import Path

__all__ = ["wait_for_file", "wait_for_mtime"]


def wait_for_file(path: Path, timeout: float, *, require_content: bool = True) -> bool:
    """Wait until ``path`` exists (and optionally has content).

    Uses :mod:`watchdog` when available.  Falls back to adaptive polling
    otherwise. Returns ``True`` if the file appeared before ``timeout``
    expires.
    """
    try:  # pragma: no cover - optional dependency
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except Exception:  # pragma: no cover - watchdog missing
        deadline = time.time() + timeout
        delay = 0.1
        while time.time() < deadline:
            if path.exists() and (not require_content or path.stat().st_size > 0):
                return True
            time.sleep(min(delay, deadline - time.time()))
            delay = min(delay * 2, 1.0)
        return False

    event = threading.Event()

    class Handler(FileSystemEventHandler):
        def _check(self, src: str) -> None:
            p = Path(src)
            if p == path and p.exists() and (
                not require_content or p.stat().st_size > 0
            ):
                event.set()

        def on_created(self, e):  # type: ignore[override]
            self._check(e.src_path)

        def on_modified(self, e):  # type: ignore[override]
            self._check(e.src_path)

    observer = Observer()
    handler = Handler()
    observer.schedule(handler, str(path.parent), recursive=False)
    observer.start()
    try:
        if path.exists() and (not require_content or path.stat().st_size > 0):
            event.set()
        event.wait(timeout)
        return event.is_set()
    finally:
        observer.stop()
        observer.join()


def wait_for_mtime(path: Path, last_mtime: float, timeout: float) -> float | None:
    """Wait for ``path`` to change modification time.

    Returns the new modification time or ``None`` if ``timeout`` expires.
    Uses :mod:`watchdog` when available and falls back to adaptive polling.
    """
    try:  # pragma: no cover - optional dependency
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except Exception:  # pragma: no cover - watchdog missing
        deadline = time.time() + timeout
        delay = 0.1
        while time.time() < deadline:
            if path.exists():
                mtime = path.stat().st_mtime
                if mtime != last_mtime:
                    return mtime
            time.sleep(min(delay, deadline - time.time()))
            delay = min(delay * 2, 1.0)
        return None

    event = threading.Event()

    class Handler(FileSystemEventHandler):
        def _maybe_set(self, src: str) -> None:
            if Path(src) == path:
                event.set()

        def on_modified(self, e):  # type: ignore[override]
            self._maybe_set(e.src_path)

        def on_created(self, e):  # type: ignore[override]
            self._maybe_set(e.src_path)

    observer = Observer()
    handler = Handler()
    observer.schedule(handler, str(path.parent), recursive=False)
    observer.start()
    try:
        if path.exists():
            m = path.stat().st_mtime
            if m != last_mtime:
                return m
        if event.wait(timeout):
            return path.stat().st_mtime if path.exists() else None
        return None
    finally:
        observer.stop()
        observer.join()
