"""H1 EMA/RSI/ATR based signal.

This module implements a simple trend–pullback strategy operating on the H1
timeframe.  The signal is composed of three building blocks:

* **Trend gate** – the absolute distance between fast and slow EMAs must exceed
  a multiple of the ATR.  This avoids ranging markets.
* **Pullback** – the previous close needs to be sufficiently close to the fast
  EMA to qualify as a pullback in the prevailing trend.
* **Trigger** – the RSI crosses the neutral 50 level in the direction of the
  trend.

Entries are placed at a breakout of the most recent high/low with an ATR buffer
and risk management also relies on ATR multiples.  Besides the trading action
additional descriptive fields of :class:`~forest5.signals.contract.TechnicalSignal`
are populated so the decision engine can consume extra context.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from forest5.core.indicators import atr, ema, rsi
from .contract import TechnicalSignal


DEFAULT_PARAMS: dict[str, Any] = {
    "ema_fast": 21,
    "ema_slow": 55,
    "atr_period": 14,
    "rsi_period": 14,
    "t_sep_atr": 0.5,  # EMA separation threshold in ATR multiples
    "pullback_atr": 0.5,  # max distance from fast EMA for pullback
    "entry_buffer_atr": 0.1,  # breakout buffer
    "sl_atr": 1.0,  # stop-loss distance in ATR multiples
    "rr": 2.0,  # risk–reward ratio
    "timeframe": "H1",
    "horizon_minutes": 240,
}


def _to_params(params: Any | None) -> dict[str, Any]:
    """Convert various parameter containers to a plain dict."""

    if params is None:
        return dict(DEFAULT_PARAMS)
    if isinstance(params, Mapping):
        cfg = {**DEFAULT_PARAMS, **params}
        return cfg
    if hasattr(params, "model_dump"):
        return {**DEFAULT_PARAMS, **params.model_dump()}  # type: ignore[attr-defined]
    if hasattr(params, "dict"):
        return {**DEFAULT_PARAMS, **params.dict()}  # type: ignore[attr-defined]
    return {**DEFAULT_PARAMS, **vars(params)}


def h1_ema_rsi_atr(df: pd.DataFrame, params: Any | None = None) -> TechnicalSignal:
    """Compute H1 EMA/RSI/ATR signal.

    Parameters
    ----------
    df:
        DataFrame with ``open``, ``high``, ``low`` and ``close`` prices.
    params:
        Dictionary-like container configuring the strategy.  Any missing keys
        fall back to :data:`DEFAULT_PARAMS`.
    """

    p = _to_params(params)

    lookback = max(p["ema_fast"], p["ema_slow"], p["atr_period"], p["rsi_period"]) + 2
    if len(df) < lookback:
        return TechnicalSignal(timeframe=p["timeframe"], horizon_minutes=p["horizon_minutes"])

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema_f = ema(close, p["ema_fast"])
    ema_s = ema(close, p["ema_slow"])
    atr_series = atr(high, low, close, p["atr_period"])
    rsi_series = rsi(close, p["rsi_period"])

    ema_f_last = ema_f.iloc[-1]
    ema_s_last = ema_s.iloc[-1]
    atr_last = atr_series.iloc[-1]
    rsi_last = rsi_series.iloc[-1]
    rsi_prev = rsi_series.iloc[-2]
    close_prev = close.iloc[-2]

    # --- Trend gate -------------------------------------------------------
    sep_ok = abs(ema_f_last - ema_s_last) >= p["t_sep_atr"] * atr_last
    trend = 1 if ema_f_last > ema_s_last else -1 if ema_f_last < ema_s_last else 0
    if not sep_ok:
        trend = 0

    # --- Pullback ---------------------------------------------------------
    ema_f_prev = ema_f.iloc[-2]
    atr_prev = atr_series.iloc[-2]
    pullback = abs(close_prev - ema_f_prev) <= p["pullback_atr"] * atr_prev

    # --- Trigger ----------------------------------------------------------
    trigger_up = rsi_prev < 50 <= rsi_last
    trigger_down = rsi_prev > 50 >= rsi_last
    trigger = (trend == 1 and trigger_up) or (trend == -1 and trigger_down)

    action = "KEEP"
    entry = sl = tp = 0.0
    drivers: list[str] = []

    if trend and pullback and trigger:
        drivers = ["ema_trend", "pullback", "rsi_trigger"]
        risk = p["sl_atr"] * atr_last
        if trend == 1:
            entry = df["high"].iloc[-1] + p["entry_buffer_atr"] * atr_last
            sl = entry - risk
            tp = entry + risk * p["rr"]
            action = "BUY"
        else:
            entry = df["low"].iloc[-1] - p["entry_buffer_atr"] * atr_last
            sl = entry + risk
            tp = entry - risk * p["rr"]
            action = "SELL"

    meta = {
        "ema_fast": float(ema_f_last),
        "ema_slow": float(ema_s_last),
        "atr": float(atr_last),
        "rsi": float(rsi_last),
    }

    technical_score = 1.0 if action == "BUY" else -1.0 if action == "SELL" else 0.0
    confidence_tech = abs(technical_score)

    return TechnicalSignal(
        timeframe=p["timeframe"],
        action=action,
        entry=entry,
        sl=sl,
        tp=tp,
        horizon_minutes=p["horizon_minutes"],
        technical_score=technical_score,
        confidence_tech=confidence_tech,
        drivers=drivers,
        meta=meta,
    )


__all__ = ["h1_ema_rsi_atr"]

