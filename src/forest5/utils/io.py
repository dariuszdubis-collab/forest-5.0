from __future__ import annotations

import csv
from pathlib import Path
import pandas as pd


def read_ohlc_csv(
    path: str | Path,
    time_col: str | None = None,
    tz: str | None = None,
    sep: str | None = None,
) -> pd.DataFrame:
    """Load OHLC CSV used by Forest5 backtests.

    This reader normalises column names (case/whitespace/aliases), parses the
    timestamp column and returns a frame indexed by a tz-naive
    :class:`~pandas.DatetimeIndex` with canonical ``open/high/low/close``
    columns.

    Parameters
    ----------
    path:
        Path to a CSV file.
    time_col:
        Name of the timestamp column. If ``None`` it will try common aliases
        such as ``time``/``timestamp``/``date``/``datetime``/``dt``.
    tz:
        Optional timezone if the input timestamps are *naive* and you know
        their timezone. They will be localized to ``tz`` and converted to
        tz-naive (wall-clock) to avoid tz-aware/naive mix-ups downstream.
    sep:
        Optional CSV separator. When ``None`` it will be auto-detected.

    Returns
    -------
    pd.DataFrame
        Data indexed by time with columns ``open``, ``high``, ``low``, ``close``.
    """

    path = Path(path)

    # detect separator if not provided
    if sep is None:
        try:
            with open(path, "r", newline="") as f:
                sample = f.read(4096)
                sep = csv.Sniffer().sniff(sample).delimiter
        except csv.Error:
            sep = ","

    df = pd.read_csv(path, sep=sep)
    # normalise headers
    df.columns = df.columns.str.strip().str.lower()

    # normalise time column argument and discover aliases when absent
    time_col = time_col.lower() if time_col is not None else None
    if time_col is None:
        for alias in ("time", "timestamp", "date", "datetime", "dt"):
            if alias in df.columns:
                time_col = alias
                break

    if time_col and time_col in df.columns:
        idx = pd.to_datetime(
            df[time_col], errors="coerce", utc=False, format="mixed"
        )
        df = df.drop(columns=[time_col])
    else:
        idx = pd.to_datetime(df.index, errors="coerce", utc=False, format="mixed")

    if idx.isna().any():
        bad = int(idx.isna().sum())
        raise ValueError(
            f"Failed to parse {bad} timestamps from '{path}'. Check the 'time' column format."
        )

    if tz is not None and getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(tz)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)

    df.index = idx

    # map OHLC aliases to canonical names
    synonyms: dict[str, list[str]] = {
        "open": ["open", "o", "op", "open_price"],
        "high": ["high", "h", "hi"],
        "low": ["low", "l", "lo"],
        "close": ["close", "c", "cl", "close_price"],
    }

    rename: dict[str, str] = {}
    for target, keys in synonyms.items():
        for k in keys:
            if k in df.columns:
                rename[k] = target
                break

    df = df.rename(columns=rename)

    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return df[need].sort_index()
