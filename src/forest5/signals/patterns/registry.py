"""Registry and utilities for candlestick patterns."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

from pandas import DataFrame

from . import engulfing, pinbar, stars

Detector = Callable[[DataFrame, float], Optional[dict]]

PATTERN_DETECTORS: Dict[str, Detector] = {
    "engulfing": engulfing.detect,
    "pinbar": pinbar.detect,
    "stars": stars.detect,
}


def best_pattern(
    df: DataFrame, atr: float, config: Optional[Mapping[str, bool]] = None
) -> Optional[dict]:
    """Return the highest score pattern based on configuration."""

    if config is not None:
        # Accept pydantic models or other containers with ``model_dump``/``dict``
        if hasattr(config, "model_dump"):
            config = config.model_dump()
        elif hasattr(config, "dict"):
            config = config.dict()

    best = None
    for name, detector in PATTERN_DETECTORS.items():
        if config is not None and config.get(name) is False:
            continue
        result = detector(df, atr)
        if not result:
            continue
        if best is None or result["score"] > best["score"]:
            best = result
    if best is None:
        return None
    # Provide generic aliases for callers expecting ``name``/``strength``
    best.setdefault("name", best.get("type"))
    best.setdefault("strength", best.get("score"))
    return best
