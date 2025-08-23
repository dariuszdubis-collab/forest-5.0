from __future__ import annotations

import csv
from pathlib import Path
import pandas as pd


def read_ohlc_csv(
    path: str | Path,
    time_col: str | None = None,
    tz: str | None = None,
    sep: str | None = None,
    has_header: bool = True,
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
    has_header:
        Set to ``False`` for files without a header row. The columns are then
        assumed to be ``time,open,high,low,close,volume`` and the first column
        is used as the index.

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

    if has_header:
        df = pd.read_csv(path, sep=sep)
        # normalise headers
        df.columns = df.columns.str.strip().str.lower()
    else:
        df = pd.read_csv(
            path,
            sep=sep,
            header=None,
            names=["time", "open", "high", "low", "close", "volume"],
            index_col=0,
        )
        df.index = pd.to_datetime(df.index, errors="coerce", utc=False, format="mixed")
        df.index.name = "time"

    # normalise time column argument and discover aliases when absent
    if has_header:
        time_col = time_col.lower() if time_col is not None else None
        if time_col is None:
            for alias in ("time", "timestamp", "date", "datetime", "dt"):
                if alias in df.columns:
                    time_col = alias
                    break

        if time_col and time_col in df.columns:
            idx = pd.to_datetime(df[time_col], errors="coerce", utc=False, format="mixed")
            df = df.drop(columns=[time_col])
        else:
            idx = pd.to_datetime(df.index, errors="coerce", utc=False, format="mixed")
    else:
        idx = df.index

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

    # map OHLCV aliases to canonical names
    synonyms: dict[str, list[str]] = {
        "open": ["open", "o", "op", "open_price"],
        "high": ["high", "h", "hi"],
        "low": ["low", "l", "lo"],
        "close": ["close", "c", "cl", "close_price"],
        # volume is optional but we try to capture common aliases
        "volume": ["volume", "vol", "v"],
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

    cols = need + (["volume"] if "volume" in df.columns else [])
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()

    return df.sort_index()


# Supported trading symbols. The list is intentionally short and can be
# expanded as the project grows.
SUPPORTED_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "EURJPY",
    "EURGBP",
]

# Default directory for historical CSV data used by helper functions.
DATA_DIR = Path("/home/daro/Fxdata")


def load_symbol_csv(symbol: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load OHLC data for ``symbol`` from ``data_dir``.

    Parameters
    ----------
    symbol:
        Trading symbol, e.g. ``"EURUSD"``. The input is upper-cased to match
        common naming conventions.
    data_dir:
        Directory containing ``<symbol>_H1.csv`` files. Defaults to
        :data:`DATA_DIR`.
        The file is inspected for a header row and the result forwarded to
        :func:`read_ohlc_csv`.

    Returns
    -------
    pd.DataFrame
        Data loaded via :func:`read_ohlc_csv`.

    Raises
    ------
    FileNotFoundError
        If the expected CSV file does not exist.
    """

    symbol = symbol.upper()
    if symbol not in SUPPORTED_SYMBOLS:
        raise ValueError(
            f"Unsupported symbol '{symbol}'. Supported symbols: {', '.join(SUPPORTED_SYMBOLS)}"
        )
    path = data_dir / f"{symbol}_H1.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV for symbol '{symbol}' not found: {path}")

    has_header = True
    try:
        with open(path, "r", newline="") as f:
            sample = f.read(4096)
            has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        pass

    return read_ohlc_csv(path, has_header=has_header)
