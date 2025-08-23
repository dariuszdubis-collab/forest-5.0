"""Signal combining EMA trend gate, pullback and RSI trigger.

This module exposes a helper returning a :class:`TechnicalSignal` describing a
trading setup on H1 timeframe.  The logic is intentionally lightweight but
captures the essence of the user's request:

* Trend gate: price must be above/below an exponential moving average (EMA)
  shifted by an Average True Range (ATR) offset.  This defines bullish and
  bearish regimes.
* Pullback detection: after a trend is established we require the previous bar
  to pull back toward the EMA.  For an uptrend the previous candle should trade
  below the EMA; for a downtrend it should trade above it.
* RSI crossing 50 trigger: once the pullback occurs, a signal is triggered when
  the Relative Strength Index (RSI) crosses the neutral 50 level in the trend
  direction.

For simplicity the entry price is the latest close.  Stop–loss and take–profit
levels are derived from the ATR (1× for SL and 2× for TP).  If conditions are
not met a neutral ``TechnicalSignal`` is returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from forest5.core.indicators import atr, ema, rsi
from .contract import TechnicalSignal


@dataclass
class Parameters:
    ema_period: int = 200
    atr_period: int = 14
    rsi_period: int = 14
    atr_offset: float = 1.0
    sl_mul: float = 1.0
    tp_mul: float = 2.0


def h1_ema_rsi_atr(df: pd.DataFrame, params: Parameters | None = None) -> TechnicalSignal:
    """Generate a signal based on EMA trend gate, pullback and RSI trigger.

    Parameters
    ----------
    df:
        DataFrame with at least ``high``, ``low`` and ``close`` columns.
    params:
        Optional tuning parameters.  Defaults mimic common trading settings.
    """

    if params is None:
        params = Parameters()

    if len(df) < 2:
        return TechnicalSignal()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema_series = ema(close, params.ema_period)
    atr_series = atr(high, low, close, params.atr_period)
    rsi_series = rsi(close, params.rsi_period)

    ema_last = ema_series.iloc[-1]
    atr_last = atr_series.iloc[-1]
    close_last = close.iloc[-1]
    close_prev = close.iloc[-2]

    rsi_last = rsi_series.iloc[-1]
    rsi_prev = rsi_series.iloc[-2]

    trend_up = close_last > ema_last + atr_last * params.atr_offset
    trend_down = close_last < ema_last - atr_last * params.atr_offset

    action = 0
    entry = sl = tp = 0.0

    if trend_up:
        pullback = close_prev < ema_series.iloc[-2]
        trigger = rsi_prev < 50 <= rsi_last
        if pullback and trigger:
            action = 1
    elif trend_down:
        pullback = close_prev > ema_series.iloc[-2]
        trigger = rsi_prev > 50 >= rsi_last
        if pullback and trigger:
            action = -1

    if action:
        entry = close_last
        if action == 1:
            sl = entry - atr_last * params.sl_mul
            tp = entry + atr_last * params.tp_mul
        else:
            sl = entry + atr_last * params.sl_mul
            tp = entry - atr_last * params.tp_mul

    return TechnicalSignal(action=action, entry=entry, sl=sl, tp=tp)
