from __future__ import annotations

import pandas as pd

from forest5.core.indicators import ema


def _ema_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Sygnał na przecięciu EMA: +1 na świecy przecięcia w górę, -1 przy przecięciu w dół, w pozostałych miejscach 0."""
    f = ema(close, fast)
    s = ema(close, slow)

    trend = f.gt(s)                      # True gdy f>s (trend wzrostowy)
    prev = trend.shift(1)                # stan z poprzedniej świecy
    cross = trend.ne(prev).fillna(False) # zmiana stanu -> przecięcie

    sig = pd.Series(0, index=close.index, dtype=int)
    sig.loc[cross & trend] = 1           # przecięcie w górę
    sig.loc[cross & ~trend] = -1         # przecięcie w dół
    return sig


def compute_signal(df: pd.DataFrame, settings, price_col: str = "close") -> pd.Series:
    """
    Zwraca serię sygnałów -1/0/+1 w zależności od wybranej strategii (settings.strategy.name).
    Domyślnie 'ema_cross' z parametrami 'fast' i 'slow' z obiektu settings.
    Dostępne:
      - 'ema_cross'
      - 'candles_engulf' (1 dla bullish engulfing, -1 dla bearish engulfing)
    """
    name = getattr(settings.strategy, "name", "ema_cross")

    if name == "ema_cross":
        fast = getattr(settings.strategy, "fast", 12)
        slow = getattr(settings.strategy, "slow", 26)
        return _ema_cross(df[price_col], fast, slow)

    elif name == "candles_engulf":
        # Import lokalny, aby nie generować ostrzeżeń linta, gdy wariant nieużywany
        from forest5.signals.candles import bullish_engulfing, bearish_engulfing

        bull = bullish_engulfing(df)
        bear = bearish_engulfing(df)

        sig = pd.Series(0, index=df.index, dtype=int)
        sig = sig.mask(bull, 1)
        sig = sig.mask(bear, -1)
        return sig.fillna(0).astype(int)

    # Fallback: brak transakcji
    return pd.Series(0, index=df.index, dtype=int)


__all__ = ["compute_signal"]

