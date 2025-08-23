from __future__ import annotations

import numpy as np
import pandas as pd

from forest5.core.indicators import ema


def macd_cross_signal(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """MACD/SIGNAL crossover signal."""
    arr = close.to_numpy(dtype=float)
    macd_arr = ema(arr, fast) - ema(arr, slow)
    sigl_arr = ema(macd_arr, signal)
    macd = pd.Series(macd_arr, index=close.index)
    sigl = pd.Series(sigl_arr, index=close.index)
    spread = macd - sigl
    sgn = np.sign(spread.fillna(0)).astype("int8")
    cross = sgn.diff().fillna(0)

    out = pd.Series(0, index=close.index, dtype="int8")
    out[(cross > 0) & (sgn > 0)] = 1
    out[(cross < 0) & (sgn < 0)] = -1
    return out
