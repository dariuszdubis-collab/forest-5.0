"""Pinbar pattern detector."""

from __future__ import annotations

import pandas as pd


def detect(
    df: pd.DataFrame,
    atr: float,
    *,
    wick_dom: float = 0.60,
    body_max: float = 0.30,
    opp_wick_max: float = 0.20,
) -> dict | None:
    """Detect bullish or bearish pinbar pattern."""

    if len(df) < 1 or atr <= 0:
        return None

    c = df.iloc[-1]
    rng = c.high - c.low
    if rng <= 0:
        return None
    body = abs(c.close - c.open)
    upper = c.high - max(c.open, c.close)
    lower = min(c.open, c.close) - c.low

    body_frac = body / rng
    upper_frac = upper / rng
    lower_frac = lower / rng

    if body_frac > body_max:
        return None

    if lower_frac >= wick_dom and upper_frac <= opp_wick_max:
        score = lower / atr
        return {
            "type": "bullish_pinbar",
            "hi": float(c.high),
            "lo": float(c.low),
            "score": float(score),
        }

    if upper_frac >= wick_dom and lower_frac <= opp_wick_max:
        score = upper / atr
        return {
            "type": "bearish_pinbar",
            "hi": float(c.high),
            "lo": float(c.low),
            "score": float(score),
        }

    return None
