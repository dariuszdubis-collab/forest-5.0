from __future__ import annotations

import numpy as np
import pandas as pd


def ema(x: pd.Series, period: int) -> pd.Series:
    """Exponential moving average of ``x`` with span ``period``."""

    return x.ewm(span=period, adjust=False, min_periods=1).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range (Wilder) based on OHLC data."""

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder: ewm z alpha=1/period i inicjalizacja pierwszą wartością TR
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()
    return atr


def atr_at_close(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """ATR computed only at bar closes.

    ``close`` may contain ``NaN`` values between closes. They are
    forward-filled for the calculation, but the returned series contains
    values only where ``close`` is present.
    """

    atr_full = atr(high, low, close.ffill(), period)
    return atr_full.where(close.notna())


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index of ``close`` prices."""

    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_cross(rsi: pd.Series, center: float = 50) -> pd.Series:
    """Return +1/-1 when ``rsi`` crosses the ``center`` level.

    +1 when RSI crosses ``center`` from below,
    -1 when it crosses from above,
    0 otherwise.
    """

    above = rsi > center
    prev_above = above.shift(1).astype(float).fillna(0.0).astype(bool)
    crossed_up = above & (~prev_above)
    crossed_down = (~above) & prev_above

    out = pd.Series(0, index=rsi.index, dtype="int8")
    out.loc[crossed_up] = 1
    out.loc[crossed_down] = -1
    return out
