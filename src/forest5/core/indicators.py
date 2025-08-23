from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, np.ndarray]


def _to_series(x: ArrayLike) -> pd.Series:
    return pd.Series(x) if isinstance(x, np.ndarray) else x


def ema(x: ArrayLike, period: int) -> ArrayLike:
    s = _to_series(x)
    result = s.ewm(span=period, adjust=False, min_periods=1).mean()
    return result.to_numpy() if isinstance(x, np.ndarray) else result


def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int) -> ArrayLike:
    high_s = _to_series(high)
    low_s = _to_series(low)
    close_s = _to_series(close)
    prev_close = close_s.shift(1)
    tr = pd.concat(
        [
            (high_s - low_s).abs(),
            (high_s - prev_close).abs(),
            (low_s - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder: ewm z alpha=1/period i inicjalizacja pierwszą wartością TR
    atr_series = tr.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()
    if (
        isinstance(high, np.ndarray)
        and isinstance(low, np.ndarray)
        and isinstance(close, np.ndarray)
    ):
        return atr_series.to_numpy()
    return atr_series


def rsi(close: ArrayLike, period: int) -> ArrayLike:
    close_s = _to_series(close)
    delta = close_s.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result.to_numpy() if isinstance(close, np.ndarray) else result


def atr_offset(value: float, k_atr: float, atr_val: float) -> float:
    return value + k_atr * atr_val
