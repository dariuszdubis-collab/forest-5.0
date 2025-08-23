from __future__ import annotations

import pandas as pd

from .factory import compute_signal
from .contract import TechnicalSignal


def contract_to_int(signal: TechnicalSignal) -> int:
    """Convert a :class:`TechnicalSignal` to ``-1``, ``0`` or ``1``.

    Any non-positive/negative action is clamped to the allowed range.
    """
    action = getattr(signal, "action", 0)
    if action > 0:
        return 1
    if action < 0:
        return -1
    return 0


def compute_signal_compat(
    df: pd.DataFrame,
    settings,
    price_col: str = "close",
) -> pd.Series:
    """Backward compatible signal computation.

    Delegates to :func:`compute_signal` and ensures the return value is a
    ``pandas.Series`` containing integer actions.
    """
    res = compute_signal(df, settings, price_col=price_col)
    if isinstance(res, pd.Series):
        return res.astype("int8")
    if isinstance(res, TechnicalSignal):
        import numpy as np

        arr = np.zeros(len(df), dtype=np.int8)
        if len(df):
            arr[-1] = contract_to_int(res)
        return pd.Series(arr, index=df.index, dtype="int8")
    # Fallback for scalar values
    return pd.Series([int(res)], index=df.index[-1:], dtype="int8")
