"""Engulfing pattern detector."""

from __future__ import annotations

import pandas as pd


def detect(df: pd.DataFrame, atr: float) -> dict | None:
    """Detect bullish or bearish engulfing pattern.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame slice containing the last two candles with columns
        ``open``, ``high``, ``low`` and ``close``.
    atr : float
        Average true range used for scoring.

    Returns
    -------
    dict | None
        A dictionary with ``type``, ``hi``, ``lo`` and ``score`` keys
        when the pattern is found otherwise ``None``.
    """
    if len(df) < 2:
        return None

    a = df.iloc[-2]
    b = df.iloc[-1]

    # Bullish engulfing
    if a.close < a.open and b.close > b.open and b.open < a.close and b.close > a.open:
        score = (abs(b.close - b.open) + abs(a.close - a.open)) / atr
        return {
            "type": "bullish_engulfing",
            "hi": float(b.high),
            "lo": float(b.low),
            "score": float(score),
        }

    # Bearish engulfing
    if a.close > a.open and b.close < b.open and b.open > a.close and b.close < a.open:
        score = (abs(b.close - b.open) + abs(a.close - a.open)) / atr
        return {
            "type": "bearish_engulfing",
            "hi": float(b.high),
            "lo": float(b.low),
            "score": float(score),
        }

    return None
