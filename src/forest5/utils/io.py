from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from ..config import get_data_dir
from .timeindex import ensure_h1

DATA_DIR = get_data_dir()

# Commonly traded forex symbols supported by the library. The check in
# ``load_symbol_csv`` ensures that we only load data for known symbols.
ALLOWED_SYMBOLS = {
    "AUDUSD",
    "EURUSD",
    "EURJPY",
    "GBPJPY",
    "GBPUSD",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
}


def sniff_csv_dialect(path: str | Path, sample_bytes: int = 65_536) -> tuple[str, str, bool]:
    """Best effort detection of CSV dialect.

    Parameters
    ----------
    path:
        CSV file to inspect.
    sample_bytes:
        Number of bytes to read for sniffing.

    Returns
    -------
    tuple
        ``(separator, decimal, has_header)`` where ``separator`` is either
        `','` or `';'`, ``decimal`` is one of ``'.'``/`','` and ``has_header``
        indicates the presence of a header row.
    """

    path = Path(path)
    try:
        with open(path, "r", newline="") as f:
            sample = f.read(sample_bytes)
    except OSError:
        return ",", ".", True

    sep = ","
    has_header = True
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";"])
        sep = dialect.delimiter
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        pass

    decimal = "."
    if sep == ";":
        # Count decimal comma vs dot occurrences in the sample
        comma_dec = len(re.findall(r"\d+,\d+", sample))
        dot_dec = len(re.findall(r"\d+\.\d+", sample))
        decimal = "," if comma_dec > dot_dec else "."

    return sep, decimal, has_header


def infer_ohlc_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, list[str]]:
    """Infer and normalise OHLC column schema.

    This maps common aliases (e.g. ``O/H/L/C`` or ``Date`` + ``Time``) to the
    canonical ``open/high/low/close/volume`` set. It also attempts to locate the
    timestamp column and returns diagnostic notes describing the transformations
    performed.
    """

    notes: list[str] = []
    df = df.copy()

    # Normalise column names for lookup while preserving originals
    cols = {c.strip().lower(): c for c in df.columns}

    time_col: str | None = None
    # Handle ``Date`` + ``Time`` split columns
    if "date" in cols and "time" in cols and "datetime" not in cols and "timestamp" not in cols:
        df["time"] = df[cols["date"]].astype(str) + " " + df[cols["time"]].astype(str)
        notes.append("combined 'date' and 'time' columns into 'time'")
        time_col = "time"
    else:
        for alias in ("time", "timestamp", "datetime", "dt"):
            if alias in cols:
                time_col = cols[alias]
                if alias != "time":
                    notes.append(f"using '{time_col}' as time column")
                break

    synonyms: dict[str, list[str]] = {
        "open": ["open", "o", "op", "open_price"],
        "high": ["high", "h", "hi"],
        "low": ["low", "l", "lo"],
        "close": ["close", "c", "cl", "close_price"],
        "volume": ["volume", "vol", "v"],
    }

    rename: dict[str, str] = {}
    for canon, aliases in synonyms.items():
        found = None
        for alias in aliases:
            if alias in cols:
                found = cols[alias]
                break
        if found is not None:
            if found != canon:
                rename[found] = canon
                notes.append(f"renamed '{found}' to '{canon}'")
        else:
            notes.append(f"missing '{canon}' column")

    if rename:
        df = df.rename(columns=rename)

    return df, time_col, notes


def read_ohlc_csv_smart(
    path: str | Path,
    time_col: str | None = None,
    sep: str | None = None,
    decimal: str | None = None,
) -> pd.DataFrame:
    """Read OHLC data with automatic dialect and schema inference."""

    path = Path(path)

    sniff_sep, sniff_dec, has_header = sniff_csv_dialect(path)
    if sep is None:
        sep = sniff_sep
    if decimal is None:
        decimal = sniff_dec

    if has_header:
        df = pd.read_csv(path, sep=sep, decimal=decimal)
    else:
        df = pd.read_csv(
            path,
            sep=sep,
            decimal=decimal,
            header=None,
            names=["time", "open", "high", "low", "close", "volume"],
        )

    df, inferred_time, notes = infer_ohlc_schema(df)

    if time_col is not None:
        lc = {c.lower(): c for c in df.columns}
        target = lc.get(time_col.lower())
        if target is None:
            raise ValueError(f"Specified time column '{time_col}' not found")
        inferred_time = target

    if inferred_time is None or inferred_time not in df.columns:
        raise ValueError("Unable to determine time column")

    idx = pd.to_datetime(df[inferred_time], errors="coerce", utc=True, format="mixed")
    if idx.isna().any():
        bad = int(idx.isna().sum())
        raise ValueError(
            f"Failed to parse {bad} timestamps from '{path}'. Check the 'time' column format."
        )

    df = df.drop(columns=[inferred_time])
    df.index = idx
    df.index.name = "time"

    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    cols = need + (["volume"] if "volume" in df.columns else [])
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_index()
    df.attrs["notes"] = notes
    return df


def read_ohlc_csv(
    path: str | Path,
    time_col: str | None = None,
    tz: str | None = None,
    sep: str | None = None,
    has_header: bool | None = None,
    decimal: str | None = None,
) -> pd.DataFrame:
    """Load OHLC CSV used by Forest5 backtests.

    Parameters
    ----------
    path:
        Path to a CSV file.
    time_col:
        Name of the timestamp column. If ``None`` it will try common aliases.
    tz:
        Optional timezone to convert the parsed index to before returning a
        tz-naive index.
    sep, has_header, decimal:
        Optional dialect overrides. When omitted the dialect is auto-detected
        via :func:`sniff_csv_dialect` and processing delegated to
        :func:`read_ohlc_csv_smart`.

    Returns
    -------
    pd.DataFrame
        Data indexed by time with canonical ``open/high/low/close`` columns.
    """

    path = Path(path)

    if has_header is None:
        df = read_ohlc_csv_smart(path, time_col=time_col, sep=sep, decimal=decimal)
    else:
        if sep is None or decimal is None:
            sniff_sep, sniff_dec, _ = sniff_csv_dialect(path)
            if sep is None:
                sep = sniff_sep
            if decimal is None:
                decimal = sniff_dec
        if has_header:
            df = pd.read_csv(path, sep=sep, decimal=decimal)
        else:
            df = pd.read_csv(
                path,
                sep=sep,
                decimal=decimal,
                header=None,
                names=["time", "open", "high", "low", "close", "volume"],
            )

        df, inferred_time, notes = infer_ohlc_schema(df)

        if time_col is not None:
            lc = {c.lower(): c for c in df.columns}
            target = lc.get(time_col.lower())
            if target is None:
                raise ValueError(f"Specified time column '{time_col}' not found")
            inferred_time = target

        if inferred_time is None or inferred_time not in df.columns:
            raise ValueError("Unable to determine time column")

        idx = pd.to_datetime(df[inferred_time], errors="coerce", utc=True, format="mixed")
        if idx.isna().any():
            bad = int(idx.isna().sum())
            raise ValueError(
                f"Failed to parse {bad} timestamps from '{path}'. Check the 'time' column format."
            )

        df = df.drop(columns=[inferred_time])
        df.index = idx
        df.index.name = "time"

        need = ["open", "high", "low", "close"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        cols = need + (["volume"] if "volume" in df.columns else [])
        df = df[cols].apply(pd.to_numeric, errors="coerce").dropna().sort_index()
        df.attrs["notes"] = notes

    if tz is not None:
        if getattr(df.index, "tz", None) is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)

    # Return tz-naive timestamps
    df.index = df.index.tz_localize(None)
    return df


def load_symbol_csv(symbol: str, data_dir: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    """Load OHLC data for ``symbol`` from ``data_dir``.

    Parameters
    ----------
    symbol:
        Trading symbol, e.g. ``"EURUSD"``. The input is upper-cased to match
        common naming conventions.
    data_dir:
        Directory containing ``<symbol>_H1.csv`` files. When ``None`` the
        location is resolved via :func:`forest5.config.get_data_dir`.
        The file is inspected for a header row and the result forwarded to
        :func:`read_ohlc_csv`.

    Returns
    -------
    tuple
        ``(df, meta)`` where ``df`` is the loaded data and ``meta`` contains
        gap information from :func:`~forest5.utils.timeindex.ensure_h1`.

    Raises
    ------
    FileNotFoundError
        If the expected CSV file does not exist.
    """

    symbol = symbol.upper()
    if symbol not in ALLOWED_SYMBOLS:
        raise ValueError(f"Unknown symbol '{symbol}'")
    data_dir_path = get_data_dir(data_dir)
    path = data_dir_path / f"{symbol}_H1.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV for symbol '{symbol}' not found: {path}")

    has_header = True
    try:
        with open(path, "r", newline="") as f:
            sample = f.read(4096)
            has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        pass

    df = read_ohlc_csv(path, has_header=has_header)
    df, meta = ensure_h1(df)
    return df, meta
