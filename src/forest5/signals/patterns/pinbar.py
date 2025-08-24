"""Pinbar pattern detector."""

from __future__ import annotations

import pandas as pd


def detect(df: pd.DataFrame, atr: float) -> dict | None:
    """Detect bullish or bearish pinbar pattern."""
    if len(df) < 1:
        return None

    c = df.iloc[-1]
    body = abs(c.close - c.open)
    upper = c.high - max(c.open, c.close)
    lower = min(c.open, c.close) - c.low

    if upper <= body and lower >= 2 * body:
        score = lower / atr
        return {
            "type": "bullish_pinbar",
            "hi": float(c.high),
            "lo": float(c.low),
            "score": float(score),
        }

    if lower <= body and upper >= 2 * body:
        score = upper / atr
        return {
            "type": "bearish_pinbar",
            "hi": float(c.high),
            "lo": float(c.low),
            "score": float(score),
        }

    return None
