from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = [
    "ema",
    "atr",
    "rsi",
    "atr_offset",
    "ensure_col",
    "ema_col_name",
    "rsi_col_name",
    "atr_col_name",
    "compute_ema",
    "compute_rsi",
    "compute_atr",
    "precompute_indicators",
]


def ensure_col(df: pd.DataFrame, name: str, compute_fn) -> pd.Series:
    if name not in df.columns:
        df[name] = compute_fn(df).astype("float32")
    return df[name]


def ema_col_name(period: int) -> str:
    return f"ema_{int(period)}"


def rsi_col_name(period: int) -> str:
    return f"rsi_{int(period)}"


def atr_col_name(period: int) -> str:
    return f"atr_{int(period)}"


def compute_ema(df: pd.DataFrame, period: int, src: str = "close") -> pd.Series:
    return df[src].ewm(span=period, adjust=False).mean()


def compute_rsi(df: pd.DataFrame, period: int, src: str = "close") -> pd.Series:
    delta = df[src].diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()


def precompute_indicators(
    df: pd.DataFrame,
    *,
    ema_periods: set[int] | None = None,
    rsi_periods: set[int] | None = None,
    atr_periods: set[int] | None = None,
    src: str = "close",
) -> list[str]:
    created: list[str] = []
    for p in sorted(ema_periods or []):
        name = ema_col_name(p)
        if name not in df.columns:
            df.loc[:, name] = compute_ema(df, p, src).astype("float32")
            created.append(name)
    for p in sorted(rsi_periods or []):
        name = rsi_col_name(p)
        if name not in df.columns:
            df.loc[:, name] = compute_rsi(df, p).astype("float32")
            created.append(name)
    for p in sorted(atr_periods or []):
        name = atr_col_name(p)
        if name not in df.columns:
            df.loc[:, name] = compute_atr(df, p).astype("float32")
            created.append(name)
    return created


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


def atr_offset(value: float, k: float) -> float:
    return value * k
