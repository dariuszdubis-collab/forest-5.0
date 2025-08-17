from __future__ import annotations

import pandas as pd

from forest5.core.indicators import ema


def macd_cross_signal(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """
    Sygnał przecięć MACD/SIGNAL.
    +1, gdy MACD przecina SIGNAL od dołu; -1, gdy od góry; 0 otherwise.
    """
    macd = ema(close, fast) - ema(close, slow)
    sigl = ema(macd, signal)
    spread = macd - sigl
    sgn = spread.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    cross = sgn.diff().fillna(0)

    out = pd.Series(0, index=close.index, dtype="int8")
    out[(cross > 0) & (sgn > 0)] = 1
    out[(cross < 0) & (sgn < 0)] = -1
    return out

