from __future__ import annotations

import pandas as pd


def ensure_backtest_ready(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    - Upewnia się, że mamy kolumny OHLC
    - Indeks typu datetime (tz-naive), posortowany, bez duplikatów
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
            df = df.set_index("time")
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'time' column.")

    # tz-naive
    if df.index.tz is not None:
        df = df.tz_convert(None)
        df.index = df.index.tz_localize(None)

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    if df[price_col].isna().any():
        df = df.dropna(subset=[price_col])

    return df

