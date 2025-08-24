"""Morning/Evening star detector."""
from __future__ import annotations

import pandas as pd


def detect(df: pd.DataFrame, atr: float) -> dict | None:
    """Detect morning star (bullish) or evening star (bearish) pattern."""
    if len(df) < 3:
        return None

    a = df.iloc[-3]
    b = df.iloc[-2]
    c = df.iloc[-1]

    body_a = abs(a.close - a.open)
    body_b = abs(b.close - b.open)
    body_c = abs(c.close - c.open)

    # Morning star
    if (
        a.close < a.open
        and body_a > body_b
        and c.close > c.open
        and c.close >= (a.open + a.close) / 2
    ):
        score = (body_a + body_c) / atr
        return {
            "type": "morning_star",
            "hi": float(max(a.high, b.high, c.high)),
            "lo": float(min(a.low, b.low, c.low)),
            "score": float(score),
        }

    # Evening star
    if (
        a.close > a.open
        and body_a > body_b
        and c.close < c.open
        and c.close <= (a.open + a.close) / 2
    ):
        score = (body_a + body_c) / atr
        return {
            "type": "evening_star",
            "hi": float(max(a.high, b.high, c.high)),
            "lo": float(min(a.low, b.low, c.low)),
            "score": float(score),
        }

    return None
