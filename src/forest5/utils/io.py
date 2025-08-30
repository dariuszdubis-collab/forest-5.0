from __future__ import annotations

import csv
import re
from pathlib import Path
import os
import tempfile
import json

from typing import Literal
import pandas as pd
import warnings

from ..config import get_data_dir
from .log import E_DATA_CSV_SCHEMA, log_event
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


H1Policy = Literal["strict", "pad", "infer", "drop"]


def _normalize_h1_index(df: pd.DataFrame, policy: H1Policy = "strict") -> pd.DataFrame:
    """Return ``df`` with a consistently hourly index."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    try:
        inferred = pd.infer_freq(df.index)
    except Exception:
        inferred = None

    if policy == "strict":
        if inferred not in ("H", "h"):
            diffs = df.index.to_series().diff().dropna()
            if (diffs != pd.Timedelta("1h")).any():
                raise ValueError("Index must have 1H frequency")
        return df
    if policy == "infer":
        if inferred not in ("H", "h"):
            raise ValueError("Index must have 1H frequency (infer)")
        return df
    if policy == "pad":
        full = pd.date_range(df.index[0], df.index[-1], freq="1h", tz=df.index.tz)
        if len(full) == len(df.index) and inferred in ("H", "h"):
            return df
        out = df.reindex(full)
        if "close" in out.columns:
            out["close"] = out["close"].ffill()
        for col in ("open", "high", "low"):
            if col in out.columns:
                out[col] = out[col].fillna(out["close"])
        if "volume" in out.columns:
            out["volume"] = out["volume"].fillna(0)
        out = out.ffill()
        return out
    if policy == "drop":
        return df.dropna()
    raise ValueError(f"Unknown H1 policy: {policy}")


def normalize_ohlc_h1(df: pd.DataFrame, policy: H1Policy = "strict") -> pd.DataFrame:
    """Public helper to normalise OHLC data to 1H frequency."""

    return _normalize_h1_index(df, policy=policy)


def atomic_to_csv(df: "pd.DataFrame", path: str | os.PathLike[str], **kwargs) -> None:
    """Write ``df`` to ``path`` atomically.

    The file is first written to a temporary location in the destination
    directory and then moved into place using :func:`os.replace`.
    """

    directory = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        tmp = f.name
        df.to_csv(f, index=False, **kwargs)
    os.replace(tmp, path)


def atomic_write_json(data: dict[str, object], path: str | os.PathLike[str]) -> None:
    """Atomically write JSON ``data`` to ``path``."""

    directory = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    os.replace(tmp, path)


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
    performed. Information about which aliases were used is stored on
    ``df.attrs['aliases']`` for later inspection.
    """

    notes: list[str] = []
    aliases_used: dict[str, str] = {}
    df = df.copy()

    # Normalise column names for lookup while preserving originals
    cols = {c.strip().lower(): c for c in df.columns}

    time_col: str | None = None
    # Handle ``Date`` + ``Time`` split columns
    if "date" in cols and "time" in cols and "datetime" not in cols and "timestamp" not in cols:
        df["time"] = df[cols["date"]].astype(str) + " " + df[cols["time"]].astype(str)
        notes.append("combined 'date' and 'time' columns into 'time'")
        time_col = "time"
        aliases_used["time"] = f"{cols['date']}+{cols['time']}"
    else:
        for alias in ("time", "timestamp", "datetime", "dt"):
            if alias in cols:
                time_col = cols[alias]
                if alias != "time":
                    notes.append(f"using '{time_col}' as time column")
                    aliases_used["time"] = cols[alias]
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
                aliases_used[canon] = found
        else:
            notes.append(f"missing '{canon}' column")

    if rename:
        df = df.rename(columns=rename)

    df.attrs["aliases"] = aliases_used
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
            available = ", ".join(df.columns)
            raise ValueError(
                f"Specified time column '{time_col}' not found (available: {available})"
            )
        inferred_time = target

    if inferred_time is None or inferred_time not in df.columns:
        cols = ", ".join(df.columns)
        raise ValueError(
            f"Unable to determine time column (found columns: {cols}). "
            f"Try --time-col or adjust --sep/--decimal."
        )

    idx = pd.to_datetime(df[inferred_time], errors="coerce", utc=True, format="mixed")
    if idx.isna().any():
        bad = int(idx.isna().sum())
        raise ValueError(
            f"Failed to parse {bad} timestamps from '{path}'. Check the 'time' column format "
            f"or specify --time-col/--sep/--decimal."
        )

    df = df.drop(columns=[inferred_time])
    df.index = idx
    df.index.name = "time"

    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        cols = ", ".join(df.columns)
        raise ValueError(f"CSV missing required columns: {missing} (found: {cols})")

    cols = need + (["volume"] if "volume" in df.columns else [])
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    df = df.sort_index()
    df.attrs["notes"] = notes

    time_from = df.index[0].isoformat() if not df.empty else None
    time_to = df.index[-1].isoformat() if not df.empty else None
    log_event(
        E_DATA_CSV_SCHEMA,
        path=str(path),
        separator=sep,
        decimal=decimal,
        has_header=has_header,
        aliases=df.attrs.get("aliases", {}),
        rows=len(df),
        **{"from": time_from, "to": time_to},
    )

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

    # Auto-detect dialect when not explicitly provided
    sniff_sep, sniff_dec, sniff_header = sniff_csv_dialect(path)
    if sep is None:
        sep = sniff_sep
    if decimal is None:
        decimal = sniff_dec
    if has_header is None:
        has_header = sniff_header

    notes: list[str] = []
    aliases: dict[str, str] = {}

    if has_header:
        # Read a small sample to infer schema and discover column aliases
        sample = pd.read_csv(path, sep=sep, decimal=decimal, nrows=100)
        sample, inferred_time, notes = infer_ohlc_schema(sample)

        if time_col is not None:
            lc = {c.lower(): c for c in sample.columns}
            target = lc.get(time_col.lower())
            if target is None:
                available = ", ".join(sample.columns)
                raise ValueError(
                    f"Specified time column '{time_col}' not found (available: {available})"
                )
            inferred_time = target

        if inferred_time is None or inferred_time not in sample.columns:
            cols = ", ".join(sample.columns)
            raise ValueError(
                f"Unable to determine time column (found columns: {cols}). "
                f"Try --time-col or adjust --sep/--decimal."
            )

        aliases = sample.attrs.get("aliases", {})
        time_alias = aliases.get("time", inferred_time)

        need = ["open", "high", "low", "close"]
        missing_need = [c for c in need if c not in sample.columns]
        if missing_need:
            cols = ", ".join(sample.columns)
            raise ValueError(f"CSV missing required columns: {missing_need} (found: {cols})")
        cols = need + (["volume"] if "volume" in sample.columns else [])
        col_aliases = {c: aliases.get(c, c) for c in cols}

        if "+" in time_alias:
            tparts = time_alias.split("+")
            parse_dates = {"time": tparts}
            usecols = tparts + list(col_aliases.values())
            index_col = "time"
        else:
            parse_dates = [time_alias]
            usecols = [time_alias] + list(col_aliases.values())
            index_col = time_alias

        # Only volume is forced to float32 to tolerate bad entries
        read_kwargs = dict(
            sep=sep,
            decimal=decimal,
            usecols=usecols,
            parse_dates=parse_dates,
            index_col=index_col,
            infer_datetime_format=True,
            date_parser=None,
            low_memory=False,
            memory_map=True,
        )
        with pd.option_context("mode.chained_assignment", None):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "The argument 'infer_datetime_format' is deprecated", FutureWarning
                )
                warnings.filterwarnings(
                    "ignore", "The argument 'date_parser' is deprecated", FutureWarning
                )
                try:
                    df = pd.read_csv(path, **read_kwargs)
                except TypeError:
                    read_kwargs.pop("date_parser", None)
                    df = pd.read_csv(path, **read_kwargs)
                except ValueError as e:
                    if "Usecols do not match" in str(e):
                        cols = ", ".join(sample.columns)
                        raise ValueError(
                            f"CSV missing required columns: {missing_need} (found: {cols})"
                        ) from None
                    raise

        rename_map = {v: k for k, v in col_aliases.items()}
        df = df.rename(columns=rename_map)
    else:
        # Headerless CSV with canonical column order
        names = ["time", "open", "high", "low", "close", "volume"]
        read_kwargs = dict(
            sep=sep,
            decimal=decimal,
            header=None,
            names=names,
            usecols=names,
            parse_dates=["time"],
            index_col="time",
            infer_datetime_format=True,
            date_parser=None,
            low_memory=False,
            memory_map=True,
        )
        with pd.option_context("mode.chained_assignment", None):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "The argument 'infer_datetime_format' is deprecated", FutureWarning
                )
                warnings.filterwarnings(
                    "ignore", "The argument 'date_parser' is deprecated", FutureWarning
                )
                try:
                    df = pd.read_csv(path, **read_kwargs)
                except TypeError:
                    read_kwargs.pop("date_parser", None)
                    df = pd.read_csv(path, **read_kwargs)

        need = ["open", "high", "low", "close"]
        cols = need + (["volume"] if "volume" in df.columns else [])

    # Validation of required columns
    missing = [c for c in ["open", "high", "low", "close"] if c not in df.columns]
    if missing:
        cols = ", ".join(df.columns)
        raise ValueError(f"CSV missing required columns: {missing} (found: {cols})")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"Failed to parse timestamps from '{path}'. Check the 'time' column format "
            f"or specify --time-col/--sep/--decimal."
        )
    if df.index.isna().any():
        bad = int(df.index.isna().sum())
        raise ValueError(
            f"Failed to parse {bad} timestamps from '{path}'. Check the 'time' column format "
            f"or specify --time-col/--sep/--decimal."
        )

    df.index.name = "time"
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float32")
        df = df[df["volume"].notna()]
    df = df[cols].sort_index()

    # Downcast volume to float32
    if "volume" in df.columns and df["volume"].dtype != "float32":
        df["volume"] = df["volume"].astype("float32")

    # Ensure UTC tz-aware index; keep existing tz if already set
    if getattr(df.index, "tz", None) is None:
        try:
            df.index = df.index.tz_localize(
                "UTC", nonexistent="shift_forward", ambiguous="NaT", errors="ignore"
            )
        except TypeError:
            df.index = df.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")

    df.attrs["notes"] = notes
    df.attrs["aliases"] = aliases

    if tz is not None:
        if getattr(df.index, "tz", None) is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)

    # Return tz-naive timestamps for backward compatibility
    df.index = df.index.tz_localize(None)
    return df


def load_symbol_csv(
    symbol: str,
    data_dir: Path | str | None = None,
    *,
    policy: str = "strict",
) -> tuple[pd.DataFrame, dict]:
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
    policy:
        Handling of missing bars passed to
        :func:`~forest5.utils.timeindex.ensure_h1`.

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
    df, meta = ensure_h1(df, policy=policy)
    df = normalize_ohlc_h1(df, policy=policy)
    return df, meta
