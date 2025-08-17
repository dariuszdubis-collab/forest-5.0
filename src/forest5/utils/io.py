from __future__ import annotations

import pandas as pd


def read_ohlc_csv(path: str, time_col: str = "time", tz: str | None = None) -> pd.DataFrame:
    """
    Load OHLC CSV used by Forest5 backtests.

    Expected columns (case-insensitive): time, open, high, low, close.
    The function lower-cases headers and returns a frame indexed by a tz-naive
    DatetimeIndex (Forest5 internals operate on tz-naive timestamps).

    Parameters
    ----------
    path : str
        Path to a CSV file.
    time_col : str, default "time"
        Name of the timestamp column. If missing, try to parse current index.
    tz : str | None
        Optional timezone if the input timestamps are *naive* and you know
        their timezone. They will be localized to `tz` and then converted to
        tz-naive (wall-clock) to avoid tz-aware/naive mix-ups downstream.

    Returns
    -------
    pd.DataFrame with columns: open, high, low, close
    """
    df = pd.read_csv(path)
    # normalize headers
    df.columns = df.columns.str.lower()

    # build index
    if time_col in df.columns:
        idx = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    else:
        idx = pd.to_datetime(df.index, errors="coerce", utc=False)

    if idx.isna().any():
        bad = int(idx.isna().sum())
        raise ValueError(
            f"Failed to parse {bad} timestamps from '{path}'. Check the 'time' column format."
        )

    # If tz is provided and the index is tz-naive, localize and then drop tz (Forest5 uses tz-naive internally)
    if tz is not None and getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(tz).tz_convert(None)

    df.index = idx
    # basic column check
    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df[need].sort_index()
