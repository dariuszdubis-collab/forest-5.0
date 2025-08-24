"""Registry and utilities for candlestick patterns."""
from __future__ import annotations

from typing import Callable, Dict, Optional

from . import engulfing, pinbar, stars

Detector = Callable[["DataFrame", float], Optional[dict]]

PATTERN_DETECTORS: Dict[str, Detector] = {
    "engulfing": engulfing.detect,
    "pinbar": pinbar.detect,
    "stars": stars.detect,
}


def best_pattern(df, atr: float, config: Optional[Dict[str, bool]] = None) -> Optional[dict]:
    """Return the highest score pattern based on configuration."""
    best = None
    for name, detector in PATTERN_DETECTORS.items():
        if config is not None and not config.get(name, False):
            continue
        result = detector(df, atr)
        if not result:
            continue
        if best is None or result["score"] > best["score"]:
            best = result
    return best
