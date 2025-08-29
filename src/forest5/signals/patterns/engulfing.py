"""Engulfing pattern detector."""

from __future__ import annotations

import pandas as pd


def detect(
    df: pd.DataFrame,
    atr: float,
    *,
    eps_atr: float = 0.05,
    body_ratio_min: float = 1.0,
) -> dict | None:
    """Detect bullish or bearish engulfing pattern.

    Parameters
    ----------
    df:
        DataFrame slice containing the last two candles with columns
        ``open``, ``high``, ``low`` and ``close``.
    atr:
        Average true range used for scoring.
    eps_atr:
        Tolerance for open/close overlap expressed in ATR multiples.
    body_ratio_min:
        Minimum ratio between the engulfing candle body and the body of the
        engulfed candle.
    """

    if len(df) < 2 or atr <= 0:
        return None

    a = df.iloc[-2]
    b = df.iloc[-1]
    eps = abs(eps_atr) * atr
    body_a = abs(a.close - a.open)
    body_b = abs(b.close - b.open)
    if body_a <= 0 or body_b <= 0:
        return None

    # Bullish engulfing
    if (
        a.close < a.open
        and b.close > b.open
        and b.open <= a.close + eps
        and b.close >= a.open - eps
        and body_b >= body_ratio_min * body_a
    ):
        score = (body_a + body_b) / atr
        return {
            "type": "bullish_engulfing",
            "hi": float(b.high),
            "lo": float(b.low),
            "score": float(score),
        }

    # Bearish engulfing
    if (
        a.close > a.open
        and b.close < b.open
        and b.open >= a.close - eps
        and b.close <= a.open + eps
        and body_b >= body_ratio_min * body_a
    ):
        score = (body_a + body_b) / atr
        return {
            "type": "bearish_engulfing",
            "hi": float(b.high),
            "lo": float(b.low),
            "score": float(score),
        }

    return None
