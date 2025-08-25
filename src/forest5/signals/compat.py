from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .factory import compute_signal
from .contract import TechnicalSignal


def contract_to_int(signal: Any) -> int:
    """Convert signal ``action`` to ``-1``, ``0`` or ``1``.

    The mapping follows::

        BUY  ->  1
        SELL -> -1
        other -> 0

    ``signal`` may be a :class:`TechnicalSignal`, a mapping, or a plain action
    value.
    """

    action: Any
    if isinstance(signal, TechnicalSignal) or hasattr(signal, "action"):
        action = getattr(signal, "action")
    elif isinstance(signal, Mapping):
        action = signal.get("action", 0)
    else:
        action = signal

    if isinstance(action, str):
        return {"BUY": 1, "SELL": -1}.get(action.upper(), 0)

    try:
        val = float(action)
    except Exception:
        return 0
    if val > 0:
        return 1
    if val < 0:
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

    import numpy as np

    if isinstance(res, (TechnicalSignal, Mapping)):
        arr = np.zeros(len(df), dtype=np.int8)
        if len(df):
            arr[-1] = contract_to_int(res)
        return pd.Series(arr, index=df.index, dtype="int8")

    return pd.Series([contract_to_int(res)], index=df.index[-1:], dtype="int8")
