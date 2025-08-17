from __future__ import annotations

import numpy as np
import pandas as pd


def ema(x: pd.Series, period: int) -> pd.Series:
    return x.ewm(span=period, adjust=False, min_periods=1).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
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


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))
