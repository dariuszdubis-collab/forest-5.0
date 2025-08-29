"""Registry and utilities for candlestick patterns."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

from pandas import DataFrame

from functools import partial

from . import engulfing, pinbar, stars

Detector = Callable[[DataFrame, float], Optional[dict]]

PATTERN_DETECTORS: Dict[str, Detector] = {
    "engulfing": engulfing.detect,
    "pinbar": pinbar.detect,
    "stars": stars.detect,
}


def enable_engulfing(*, eps_atr: float | None = 0.05, body_ratio_min: float | None = 1.0) -> None:
    """Register engulfing detector with custom thresholds."""

    if eps_atr is not None and eps_atr < 0:
        raise ValueError("eps_atr must be >= 0")
    if body_ratio_min is not None and body_ratio_min < 0:
        raise ValueError("body_ratio_min must be >= 0")
    PATTERN_DETECTORS["engulfing"] = partial(
        engulfing.detect,
        eps_atr=eps_atr if eps_atr is not None else 0.0,
        body_ratio_min=body_ratio_min if body_ratio_min is not None else 1.0,
    )


def enable_pinbar(
    *,
    wick_dom: float | None = 0.60,
    body_max: float | None = 0.30,
    opp_wick_max: float | None = 0.20,
) -> None:
    """Register pinbar detector with custom thresholds."""

    for name, val in {
        "wick_dom": wick_dom,
        "body_max": body_max,
        "opp_wick_max": opp_wick_max,
    }.items():
        if val is not None and val < 0:
            raise ValueError(f"{name} must be >= 0")
    PATTERN_DETECTORS["pinbar"] = partial(
        pinbar.detect,
        wick_dom=wick_dom if wick_dom is not None else 0.60,
        body_max=body_max if body_max is not None else 0.30,
        opp_wick_max=opp_wick_max if opp_wick_max is not None else 0.20,
    )


def enable_stars(*, reclaim_min: float | None = 0.62, mid_small_max: float | None = 0.40) -> None:
    """Register morning/evening star detector with custom thresholds."""

    if reclaim_min is not None and reclaim_min < 0:
        raise ValueError("reclaim_min must be >= 0")
    if mid_small_max is not None and mid_small_max < 0:
        raise ValueError("mid_small_max must be >= 0")
    PATTERN_DETECTORS["stars"] = partial(
        stars.detect,
        reclaim_min=reclaim_min if reclaim_min is not None else 0.62,
        mid_small_max=mid_small_max if mid_small_max is not None else 0.40,
    )


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
