from __future__ import annotations

import pandas as pd


def apply_rsi_filter(
    base_signal: pd.Series,
    rsi: pd.Series,
    overbought: float,
    oversold: float,
) -> pd.Series:
    """
    Jeżeli RSI >= overbought -> siła do sprzedaży (-1),
    Jeżeli RSI <= oversold   -> siła do kupna (+1),
    W przeciwnym wypadku zostawiamy bazowy sygnał.
    """
    out = base_signal.copy()
    out[rsi >= overbought] = -1
    out[rsi <= oversold] = 1
    return out.astype("int8")


def confirm_with_candles(base_signal: pd.Series, candles: pd.Series) -> pd.Series:
    """
    Gdy świeca sygnalizuje kierunek przeciwny do bazowego – zerujemy sygnał.
    Gdy zgodny – zostawiamy bazowy. Gdy 0 – bez zmian.
      candles: +1 / 0 / -1
    """
    out = base_signal.copy()
    conflict = (candles * base_signal) < 0
    out[conflict] = 0
    return out.astype("int8")
