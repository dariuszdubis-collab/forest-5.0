"""Morning/Evening star detector."""

from __future__ import annotations

import pandas as pd


def detect(
    df: pd.DataFrame,
    atr: float,
    *,
    reclaim_min: float = 0.62,
    mid_small_max: float = 0.40,
) -> dict | None:
    """Detect morning star (bullish) or evening star (bearish) pattern."""

    if len(df) < 3 or atr <= 0:
        return None

    a = df.iloc[-3]
    b = df.iloc[-2]
    c = df.iloc[-1]

    body_a = abs(a.close - a.open)
    body_b = abs(b.close - b.open)
    body_c = abs(c.close - c.open)
    if body_a <= 0:
        return None

    mid_ratio = body_b / body_a

    # Morning star
    if a.close < a.open and c.close > c.open:
        reclaim = (c.close - a.close) / body_a
        if reclaim >= reclaim_min and mid_ratio <= mid_small_max:
            score = (body_a + body_c) / atr
            return {
                "type": "morning_star",
                "hi": float(max(a.high, b.high, c.high)),
                "lo": float(min(a.low, b.low, c.low)),
                "score": float(score),
            }

    # Evening star
    if a.close > a.open and c.close < c.open:
        reclaim = (a.close - c.close) / body_a
        if reclaim >= reclaim_min and mid_ratio <= mid_small_max:
            score = (body_a + body_c) / atr
            return {
                "type": "evening_star",
                "hi": float(max(a.high, b.high, c.high)),
                "lo": float(min(a.low, b.low, c.low)),
                "score": float(score),
            }

    return None
