from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.indicators import ema


def _ema_cross_signal(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    Generuje impuls BUY/SELL tylko na realnej zmianie kierunku (cross),
    nigdy na pierwszym barze.
    """
    f = ema(close, fast)
    s = ema(close, slow)
    direction = pd.Series(np.where(f > s, 1, -1), index=close.index, dtype=int)

    # Zmiana tylko gdy mamy poprzednią wartość (brak sygnału na barze 0)
    changed = direction.ne(direction.shift(1)) & direction.shift(1).notna()

    out = pd.Series(0, index=close.index, dtype=int)
    out.loc[changed] = direction.loc[changed]
    return out


def compute_signal(df: pd.DataFrame, settings, price_col: str = "close") -> pd.Series:
    name = getattr(settings.strategy, "name", "ema_cross")
    if name in {"ema_rsi", "ema-cross+rsi"}:
        settings.strategy.name = "ema_cross"
        settings.strategy.use_rsi = True
        name = "ema_cross"
    if name == "ema_cross":
        return _ema_cross_signal(df[price_col], settings.strategy.fast, settings.strategy.slow)
    raise ValueError(f"Unknown strategy: {name}")
