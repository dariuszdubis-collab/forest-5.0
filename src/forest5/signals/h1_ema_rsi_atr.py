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
import types

import pandas as pd

from forest5.core.indicators import (
    ensure_col,
    ema_col_name,
    rsi_col_name,
    atr_col_name,
    ema,
    atr,
    rsi,
)
from .contract import TechnicalSignal
from .setups import SetupRegistry
from forest5.utils.log import TelemetryContext
from forest5.utils.debugger import TraceCollector
from . import patterns


DEFAULT_PARAMS: dict[str, Any] = {
    "ema_fast": 21,
    "ema_slow": 55,
    "atr_period": 14,
    "rsi_period": 14,
    "t_sep_atr": 0.5,  # EMA separation threshold in ATR multiples
    "pullback_atr": 0.5,  # max distance from fast EMA for pullback
    "entry_buffer_atr": 0.1,  # breakout buffer
    "sl_atr": 1.0,  # stop-loss distance in ATR multiples
    "sl_min_atr": 0.0,  # minimum stop-loss distance in ATR multiples
    "rr": 2.0,  # risk–reward ratio
    "trailing_atr": 0.0,  # optional trailing stop in ATR multiples (0=off)
    "timeframe": "H1",
    "horizon_minutes": 240,
    "ttl_minutes": None,
    "engulf_eps_atr": 0.05,
    "engulf_body_ratio_min": 1.0,
    "pinbar_wick_dom": 0.60,
    "pinbar_body_max": 0.30,
    "pinbar_opp_wick_max": 0.20,
    "star_reclaim_min": 0.62,
    "star_mid_small_max": 0.40,
    "enable_engulf": True,
    "enable_pinbar": True,
    "enable_star": True,
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


def compute_primary_signal_h1(
    df,
    params: Any | None = None,
    registry: SetupRegistry | None = None,
    ctx: TelemetryContext | None = None,
    collector: TraceCollector | None = None,
):
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
    reg = registry or SetupRegistry()

    cfg = types.SimpleNamespace(**p)
    if cfg.enable_engulf:
        patterns.registry.enable_engulfing(
            eps_atr=cfg.engulf_eps_atr,
            body_ratio_min=cfg.engulf_body_ratio_min,
        )
    if cfg.enable_pinbar:
        patterns.registry.enable_pinbar(
            wick_dom=cfg.pinbar_wick_dom,
            body_max=cfg.pinbar_body_max,
            opp_wick_max=cfg.pinbar_opp_wick_max,
        )
    if cfg.enable_star:
        patterns.registry.enable_stars(
            reclaim_min=cfg.star_reclaim_min,
            mid_small_max=cfg.star_mid_small_max,
        )

    if df.empty:
        return TechnicalSignal(
            timeframe=p["timeframe"],
            horizon_minutes=p["horizon_minutes"],
            ttl_minutes=p.get("ttl_minutes"),
            technical_score=0.0,
            confidence_tech=0.0,
            drivers=[],
            meta={},
        )

    idx = len(df) - 1
    high = df["high"]
    low = df["low"]

    now = df.index[idx]
    if not isinstance(now, pd.Timestamp):
        now = pd.Timestamp(now)
    triggered = reg.check(index=idx, price=float(df["high"].iloc[-1]), now=now, ctx=ctx)
    if not triggered:
        triggered = reg.check(index=idx, price=float(df["low"].iloc[-1]), now=now, ctx=ctx)
    if triggered:
        if collector:
            collector.note("setup_trigger", "ok", at=now, extras={"action": triggered.action})
        return triggered

    lookback = max(p["ema_fast"], p["ema_slow"], p["atr_period"], p["rsi_period"]) + 2
    if len(df) < lookback:
        return TechnicalSignal(
            timeframe=p["timeframe"],
            horizon_minutes=p["horizon_minutes"],
            ttl_minutes=p.get("ttl_minutes"),
            technical_score=0.0,
            confidence_tech=0.0,
            drivers=[],
            meta={},
        )

    close = df["close"]

    fast_name = ema_col_name(p["ema_fast"])
    slow_name = ema_col_name(p["ema_slow"])
    atr_name = atr_col_name(p["atr_period"])
    rsi_name = rsi_col_name(p["rsi_period"])

    ema_f = (
        df[fast_name]
        if fast_name in df.columns
        else ensure_col(df, fast_name, lambda d: ema(d["close"], p["ema_fast"]))
    )
    ema_s = (
        df[slow_name]
        if slow_name in df.columns
        else ensure_col(df, slow_name, lambda d: ema(d["close"], p["ema_slow"]))
    )
    atr_series = (
        df[atr_name]
        if atr_name in df.columns
        else ensure_col(
            df, atr_name, lambda d: atr(d["high"], d["low"], d["close"], p["atr_period"])
        )
    )
    rsi_series = (
        df[rsi_name]
        if rsi_name in df.columns
        else ensure_col(df, rsi_name, lambda d: rsi(d["close"], p["rsi_period"]))
    )

    ema_f_last = ema_f.iloc[-1]
    ema_s_last = ema_s.iloc[-1]
    atr_last = atr_series.iloc[-1]
    rsi_last = rsi_series.iloc[-1]
    rsi_prev = rsi_series.iloc[-2]
    close_prev = close.iloc[-2]

    meta = {
        "ema_fast": float(ema_f_last),
        "ema_slow": float(ema_s_last),
        "atr": float(atr_last),
        "rsi": float(rsi_last),
    }
    if collector:
        collector.note("setup_candidate", "base_ok", at=now, extras=meta)

    # --- Trend & pullback gates (optional) -------------------------------
    use_ema_gates = bool(p.get("use_ema_gates", True))
    if use_ema_gates:
        sep_ok = abs(ema_f_last - ema_s_last) >= p["t_sep_atr"] * atr_last
        trend = 1 if ema_f_last > ema_s_last else -1 if ema_f_last < ema_s_last else 0
        if not sep_ok:
            trend = 0
            if collector:
                collector.note("setup_candidate", "trend_gate_failed", at=now, extras=meta)

        ema_f_prev = ema_f.iloc[-2]
        atr_prev = atr_series.iloc[-2]
        pullback = abs(close_prev - ema_f_prev) <= p["pullback_atr"] * atr_prev
        if trend and not pullback and collector:
            collector.note("setup_candidate", "pullback_gate_failed", at=now, extras=meta)
    else:
        # Without EMA gates, derive trend from RSI cross direction only when it occurs
        trend = 0
        pullback = True

    # --- Trigger ----------------------------------------------------------
    trigger_up = rsi_prev < 50 <= rsi_last
    trigger_down = rsi_prev > 50 >= rsi_last
    trigger = (trend == 1 and trigger_up) or (trend == -1 and trigger_down)
    if not use_ema_gates and not trigger:
        # No RSI cross -> no opportunity without EMA gating
        pass
    if not use_ema_gates and trigger and trend == 0:
        trend = 1 if trigger_up else -1
    if trend and pullback and not trigger and collector:
        collector.note("setup_candidate", "rsi_gate_blocked", at=now, extras=meta)

    drivers: list[Any] = []
    action = "KEEP"
    entry = sl = tp = 0.0

    if trend and pullback and trigger:
        drivers = ["ema_trend", "pullback", "rsi_trigger"]
        risk = max(p["sl_atr"], p.get("sl_min_atr", 0.0)) * atr_last
        mode = p.get("entry_mode", "breakout")
        if mode == "breakout":
            if trend == 1:
                entry = high.iloc[-1] + p["entry_buffer_atr"] * atr_last
                sl = entry - risk
                tp = entry + risk * p["rr"]
                action = "BUY"
            else:
                entry = low.iloc[-1] - p["entry_buffer_atr"] * atr_last
                sl = entry + risk
                tp = entry - risk * p["rr"]
                action = "SELL"
        else:
            # entry on close +/- buffer (this bar) or force next-open trigger
            close_last = float(close.iloc[-1])
            if trend == 1:
                entry = close_last + p["entry_buffer_atr"] * atr_last
                sl = entry - risk
                tp = entry + risk * p["rr"]
                action = "BUY"
            else:
                entry = close_last - p["entry_buffer_atr"] * atr_last
                sl = entry + risk
                tp = entry - risk * p["rr"]
                action = "SELL"
            if mode == "close_next":
                # Hint to engine to trigger on next open regardless of level
                meta["entry_on_next_open"] = True

        technical_score = 1.0 if action == "BUY" else -1.0
        confidence_tech = abs(technical_score)

        patterns_cfg = dict(p.get("patterns", {}) or {})
        patterns_cfg.update(
            {
                "engulfing": cfg.enable_engulf,
                "pinbar": cfg.enable_pinbar,
                "stars": cfg.enable_star,
            }
        )
        if cfg.enable_engulf or cfg.enable_pinbar or cfg.enable_star:
            patterns_cfg["enabled"] = True

        pattern_ok = True
        if patterns_cfg.get("enabled"):
            pattern_ok = False
            pattern = patterns.registry.best_pattern(df, atr_last, patterns_cfg)
            if pattern:
                name = pattern.get("name") or pattern.get("type")
                strength = pattern.get("strength", pattern.get("score", 0.0))
                if strength >= patterns_cfg.get("min_strength", 0.0):
                    confidence_tech = max(
                        0.0, min(1.0, confidence_tech + patterns_cfg.get("boost_conf", 0.0))
                    )
                    boost_score = patterns_cfg.get("boost_score", 0.0)
                    technical_score += boost_score if technical_score > 0 else -boost_score
                    drivers.append({"pattern": name, "strength": strength})
                    meta["pattern"] = name
                    pattern_ok = True
        if collector:
            reason = "pattern_trigger_hit" if pattern_ok else "pattern_trigger_miss"
            collector.note("pattern", reason, at=now, extras=meta)
        if patterns_cfg.get("gate") and not pattern_ok:
            return TechnicalSignal(
                timeframe=p["timeframe"],
                horizon_minutes=p["horizon_minutes"],
                ttl_minutes=p.get("ttl_minutes"),
                technical_score=0.0,
                confidence_tech=0.0,
                drivers=[],
                meta=meta,
            )
        # Include trailing_atr hint in meta so engine can activate trailing stops
        if p.get("trailing_atr", 0.0):
            meta["trailing_atr"] = float(p["trailing_atr"])

        signal = TechnicalSignal(
            timeframe=p["timeframe"],
            action=action,
            entry=entry,
            sl=sl,
            tp=tp,
            horizon_minutes=p["horizon_minutes"],
            ttl_minutes=p.get("ttl_minutes"),
            technical_score=technical_score,
            confidence_tech=confidence_tech,
            drivers=drivers,
            meta=meta,
        )
        reg.arm(
            p["timeframe"], idx, signal, bar_time=now, ttl_minutes=p.get("ttl_minutes"), ctx=ctx
        )
        if collector:
            extras = {
                "action": action,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "drivers": drivers,
            }
            extras.update(meta)
            collector.note("setup_trigger", "armed", at=now, extras=extras)

    return TechnicalSignal(
        timeframe=p["timeframe"],
        horizon_minutes=p["horizon_minutes"],
        ttl_minutes=p.get("ttl_minutes"),
        technical_score=0.0,
        confidence_tech=0.0,
        drivers=[],
        meta=meta,
    )


__all__ = ["compute_primary_signal_h1"]
