from __future__ import annotations

import pandas as pd


def ensure_backtest_ready(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    - Upewnia się, że mamy kolumny OHLC
    - Ustawia indeks czasu (tz-naive) z aliasów: time/date/datetime/timestamp
    - Sortuje, usuwa duplikaty, nazywa indeks 'time'
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    # Ustaw indeks czasu jeżeli nie jest datetimem
    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = None
        for alias in ("time", "date", "datetime", "timestamp"):
            if alias in df.columns:
                time_col = alias
                break
        if time_col is None:
            raise ValueError("DataFrame must have DatetimeIndex or one of: 'time', 'date', 'datetime', 'timestamp'.")
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], utc=False, errors="coerce")
        df = df.set_index(time_col)

    # Normalizacja indeksu czasu
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index.name = "time"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Upewnij się, że OHLC są numeryczne
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    return df

