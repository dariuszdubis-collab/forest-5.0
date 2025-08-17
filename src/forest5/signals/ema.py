from __future__ import annotations

import numpy as np
import pandas as pd

from forest5.core.indicators import ema


def ema_cross_signal(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    Sygnał krzyżowania EMA.
    +1 gdy EMA(fast) > EMA(slow) i nastąpiło przebicie od dołu,
    -1 gdy EMA(fast) < EMA(slow) i nastąpiło przebicie od góry,
     0 w pozostałych punktach.
    """
    fast_ = ema(close, fast)
    slow_ = ema(close, slow)
    spread = fast_ - slow_

    # detekcja przecięć (zmiana znaku spreadu)
    sign = np.sign(spread)
    cross = sign.diff().fillna(0)

    long_sig = (cross > 0) & (sign > 0)     # przejście na dodatnie
    short_sig = (cross < 0) & (sign < 0)    # przejście na ujemne

    sig = pd.Series(0, index=close.index, dtype="int8")
    sig[long_sig] = 1
    sig[short_sig] = -1
    return sig

