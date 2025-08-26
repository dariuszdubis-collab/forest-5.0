from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class GapInfo:
    """Information about a gap in a :class:`pandas.DatetimeIndex`.

    Attributes
    ----------
    start:
        Timestamp before the gap (UTC).
    end:
        Timestamp after the gap (UTC).
    missing:
        Number of missing expected periods between ``start`` and ``end``.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    missing: int


def _to_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return ``idx`` converted to UTC and sorted."""
    idx = pd.DatetimeIndex(idx).sort_values()
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def report_gaps(idx: pd.DatetimeIndex, expected: str = "1h") -> List[GapInfo]:
    """Return information about gaps greater than ``expected``.

    ``idx`` is first converted to UTC to avoid false positives around DST
    transitions.
    """

    idx_utc = _to_utc(idx)
    step = pd.Timedelta(expected)
    gaps: List[GapInfo] = []
    for prev, cur in zip(idx_utc[:-1], idx_utc[1:]):
        delta = cur - prev
        if delta > step:
            missing = int(delta / step) - 1
            gaps.append(GapInfo(start=prev, end=cur, missing=missing))
    return gaps


def ensure_h1(df: pd.DataFrame, policy: str = "strict") -> Tuple[pd.DataFrame, dict]:
    """Ensure ``df`` has a 1-hourly index.

    Parameters
    ----------
    df:
        Data indexed by :class:`pandas.DatetimeIndex`.
    policy:
        ``'strict'`` raises :class:`ValueError` if the index is irregular.
        ``'pad'`` inserts missing rows with ``NaN`` values. ``'drop'`` removes
        rows corresponding to missing periods.

    Returns
    -------
    tuple
        ``(df_out, meta)`` where ``meta['gaps']`` contains a list of
        :class:`GapInfo` entries describing gaps greater than 1 hour.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex with 1H step")

    if df.empty:
        return df.copy(), {"gaps": []}

    df = df.copy()
    idx_utc = _to_utc(df.index)
    step = pd.Timedelta("1h")
    deltas = pd.Series(idx_utc[1:] - idx_utc[:-1])
    irregular = (deltas != step).any()
    gaps = report_gaps(idx_utc, "1h")
    median_minutes = (
        float(deltas.median() / pd.Timedelta(minutes=1)) if not deltas.empty else 60.0
    )

    if irregular:
        if policy == "strict":
            raise ValueError("Index must have 1H frequency")
        full = pd.date_range(idx_utc[0], idx_utc[-1], freq=step, tz="UTC")
        df.index = idx_utc
        df = df[~df.index.duplicated(keep="first")]
        df = df.reindex(full)
        if policy == "drop":
            df = df.dropna()
    else:
        df.index = idx_utc

    df.index = df.index.tz_localize(None)
    df.index.name = "time"
    return df, {"gaps": gaps, "median_bar_minutes": median_minutes}
