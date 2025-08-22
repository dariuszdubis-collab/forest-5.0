from __future__ import annotations

import numpy as np
import pandas as pd

from forest5.core.indicators import ema


def ema_cross_signal(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Generuje impuls BUY/SELL na realnej zmianie kierunku krzyżowania EMA."""
    fast_ = ema(close, fast)
    slow_ = ema(close, slow)
    direction = pd.Series(np.where(fast_ > slow_, 1, -1), index=close.index, dtype=int)

    # Brak sygnału na pierwszym barze — wymagamy poprzedniej wartości
    changed = direction.ne(direction.shift(1)) & direction.shift(1).notna()

    out = pd.Series(0, index=close.index, dtype=int)
    out.loc[changed] = direction.loc[changed]
    return out
