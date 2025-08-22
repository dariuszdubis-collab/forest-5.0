from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import BacktestSettings
from ..utils.debugger import DebugLogger
from ..core.indicators import atr, ema, rsi
from ..utils.log import setup_logger
from ..utils.validate import ensure_backtest_ready
from forest5.signals.factory import compute_signal
from .risk import RiskManager
from .tradebook import TradeBook
from ..time_only import TimeOnlyModel
from ..signals.fusion import _to_sign


log = setup_logger()


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    max_dd: float
    trades: TradeBook


def _validate_data(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Zapewnia poprawność danych wejściowych do backtestu."""
    return ensure_backtest_ready(df, price_col=price_col).copy()


def _generate_signal(df: pd.DataFrame, settings: BacktestSettings, price_col: str) -> pd.Series:
    """Generuje serię sygnałów tradingowych."""
    name = getattr(settings.strategy, "name", "ema_cross")
    use_rsi = getattr(settings.strategy, "use_rsi", False)
    if name in {"ema_rsi", "ema-cross+rsi"}:
        use_rsi = True

    sig = compute_signal(df, settings, price_col=price_col).astype(int)
    if use_rsi:
        rr = rsi(df[price_col], settings.strategy.rsi_period)
        sig = sig.where(~rr.ge(settings.strategy.rsi_overbought), other=-1)
        sig = sig.where(~rr.le(settings.strategy.rsi_oversold), other=1)
    return sig


def bootstrap_position(
    df: pd.DataFrame,
    sig: pd.Series,
    rm: RiskManager,
    tb: TradeBook,
    settings: BacktestSettings,
    price_col: str,
    atr_multiple: float,
) -> float:
    """Próba otwarcia pozycji na pierwszym barze dla skrajnych parametrów EMA."""
    position = 0.0
    if len(df) > 0 and settings.strategy.fast <= 2 and settings.strategy.slow >= 50:
        f0 = ema(df[price_col], settings.strategy.fast)
        s0 = ema(df[price_col], settings.strategy.slow)
        first_sig = int(sig.iloc[0]) if len(sig) else 0
        if first_sig == 0 and float(f0.iloc[0]) >= float(s0.iloc[0]):
            p0 = float(df[price_col].iloc[0])
            a0 = float(df["atr"].iloc[0]) if pd.notna(df["atr"].iloc[0]) else 0.0
            qty0 = rm.position_size(price=p0, atr=a0, atr_multiple=atr_multiple)
            if qty0 > 0.0:
                rm.buy(p0, qty0)
                tb.add(df.index[0], p0, qty0, "BUY")
                position += qty0
    return position


def _fuse_with_time(
    tech_signal: int,
    ts: pd.Timestamp,
    price: float,
    time_model: TimeOnlyModel | None,
    min_conf: int,
    ai_decision: int | None = None,
    *,
    return_reason: bool = False,
) -> int | tuple[int, str | None]:
    """Fuse technical, time and optional AI signals into one decision.

    When ``return_reason`` is True, a tuple of ``(decision, reason)`` is
    returned where ``reason`` is ``None`` unless the decision is to skip
    due to lack of confluence or a WAIT from the time model.
    """

    votes = {"tech": _to_sign(tech_signal), "time": 0, "ai": 0}
    pos = {"tech": int(votes["tech"] > 0), "time": 0, "ai": 0}
    neg = {"tech": int(votes["tech"] < 0), "time": 0, "ai": 0}

    if time_model:
        tm_decision = time_model.decide(ts, price)
        if tm_decision == "WAIT":
            res = (0, "time_model_wait") if return_reason else 0
            return res
        votes["time"] = _to_sign(1 if tm_decision == "BUY" else -1)
        pos["time"] = int(votes["time"] > 0)
        neg["time"] = int(votes["time"] < 0)

    if ai_decision is not None:
        votes["ai"] = _to_sign(ai_decision)
        pos["ai"] = int(votes["ai"] > 0)
        neg["ai"] = int(votes["ai"] < 0)

    pos_total = sum(pos.values())
    neg_total = sum(neg.values())
    if max(pos_total, neg_total) < max(min_conf, 1):
        res = (0, "not_enough_confluence") if return_reason else 0
        return res
    if pos_total > neg_total:
        res_dec = 1
    elif neg_total > pos_total:
        res_dec = -1
    else:
        res_dec = 0
    if return_reason:
        return res_dec, None
    return res_dec


def _trading_loop(
    df: pd.DataFrame,
    sig: pd.Series,
    rm: RiskManager,
    tb: TradeBook,
    position: float,
    price_col: str,
    atr_multiple: float,
    settings: BacktestSettings,
    debug: DebugLogger | None = None,
) -> float:
    """Główna pętla tradingowa z mark-to-market."""
    prices = df[price_col].to_numpy(dtype=float)
    atr_vals = df["atr"].to_numpy(dtype=float)
    sig_vals = sig.reindex(df.index).fillna(0).to_numpy(dtype=int)

    time_model: TimeOnlyModel | None = None
    if settings.time.model.enabled and settings.time.model.path:
        time_model = TimeOnlyModel.load(settings.time.model.path)

    for t, price, atr_val, this_sig in zip(df.index, prices, atr_vals, sig_vals):
        this_sig = int(this_sig)

        fused, fuse_reason = _fuse_with_time(
            this_sig,
            t,
            float(price),
            time_model,
            settings.time.fusion_min_confluence,
            None,
            return_reason=True,
        )
        this_sig = fused
        if fuse_reason == "time_model_wait" and debug:
            debug.log("skip_candle", time=str(t), reason="time_model_wait")
        if fuse_reason == "not_enough_confluence" and debug:
            debug.log("signal_rejected", time=str(t), reason="no_confluence")

        if this_sig < 0 and position > 0.0:
            rm.sell(price, position)
            tb.add(t, price, position, "SELL")
            if debug:
                debug.log("position_close", time=str(t), price=float(price), qty=float(position))
            position = 0.0

        if this_sig > 0 and position <= 0.0:
            qty = rm.position_size(price=price, atr=float(atr_val), atr_multiple=atr_multiple)
            if qty > 0.0:
                rm.buy(price, qty)
                tb.add(t, price, qty, "BUY")
                if debug:
                    debug.log("position_open", time=str(t), price=float(price), qty=float(qty))
                position = qty
            elif debug:
                debug.log(
                    "signal_rejected",
                    time=str(t),
                    reason="qty_zero",
                    price=float(price),
                    atr=float(atr_val),
                )

        equity_mtm = rm.equity + position * price
        rm.record_mark_to_market(equity_mtm)

        if rm.exceeded_max_dd():
            log.warning("max_dd_exceeded", time=str(t), equity=equity_mtm)
            break
    return position


def _compute_metrics(equity_curve: list[float]) -> tuple[pd.Series, float]:
    """Wylicza końcową krzywą equity i maksymalny drawdown."""
    eq = pd.Series(equity_curve, dtype=float)
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0, np.nan)
    max_dd = float(dd.max(skipna=True)) if len(dd) else 0.0
    return eq, max_dd


def run_backtest(
    df: pd.DataFrame,
    settings: BacktestSettings,
    risk: Optional[RiskManager] = None,
    symbol: str = "SYMBOL",
    price_col: str = "close",
    atr_period: int | None = None,
    atr_multiple: float | None = None,
) -> BacktestResult:
    # 1) walidacja danych
    df = _validate_data(df, price_col=price_col)

    # 2) generowanie sygnału
    sig = _generate_signal(df, settings, price_col=price_col)

    # 3) przygotowanie ATR do sizingu
    ap = int(atr_period or settings.atr_period)
    am = float(atr_multiple or settings.atr_multiple)
    df["atr"] = atr(df["high"], df["low"], df["close"], ap)

    # 4) stan początkowy
    tb = TradeBook()
    rm = risk or RiskManager(**settings.risk.model_dump())

    # 5) bootstrap potencjalnej pozycji
    position = bootstrap_position(df, sig, rm, tb, settings, price_col, am)

    # 6) główna pętla handlowa
    debug = DebugLogger(settings.debug_dir) if settings.debug_dir else None
    try:
        position = _trading_loop(
            df,
            sig,
            rm,
            tb,
            position,
            price_col,
            am,
            settings,
            debug,
        )
    finally:
        if debug:
            debug.close()

    # 7) metryki
    eq, max_dd = _compute_metrics(rm.equity_curve)
    return BacktestResult(equity_curve=eq, max_dd=max_dd, trades=tb)
