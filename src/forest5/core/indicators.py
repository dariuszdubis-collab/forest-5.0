from __future__ import annotations

import numpy as np
import pandas as pd


def _to_series(x) -> pd.Series:
    """Ensure input is a float pandas Series."""
    return pd.Series(np.asarray(x, dtype=float))


def ema(x, period: int) -> np.ndarray:
    """Exponential moving average for a 1-D array."""
    s = _to_series(x)
    return s.ewm(span=period, adjust=False, min_periods=1).mean().to_numpy()


def atr(high, low, close, period: int) -> np.ndarray:
    """Average True Range for 1-D arrays."""
    h = _to_series(high)
    l = _to_series(low)
    c = _to_series(close)
    prev_close = c.shift(1)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()
    return atr.to_numpy()


def rsi(close, period: int) -> np.ndarray:
    """Relative Strength Index for a 1-D array."""
    c = _to_series(close)
    delta = c.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).to_numpy()


def atr_offset(value: float, k_atr: float, atr_val: float) -> float:
    """Offset a value by ``k_atr`` multiples of ``atr_val``."""
    return float(value) + float(k_atr) * float(atr_val)
