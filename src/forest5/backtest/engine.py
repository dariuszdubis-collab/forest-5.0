from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from ..config import BacktestSettings
from ..core.indicators import atr, ema
from ..utils.log import setup_logger
from ..utils.validate import ensure_backtest_ready
from ..utils.timeframes import _TF_MINUTES
from forest5.signals.contract import TechnicalSignal
from forest5.signals.factory import compute_signal
from forest5.signals.setups import SetupRegistry
from ..time_only import TimeOnlyModel
from .risk import RiskManager
from .tradebook import TradeBook
from ..signals.fusion import _to_sign


log = setup_logger()


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    max_dd: float
    trades: TradeBook


class TPslPolicy(str, Enum):
    """Priority rule when both TP and SL are hit on the same bar."""

    TP = "tp"
    SL = "sl"


@dataclass
class Position:
    qty: float
    entry: float
    sl: float
    tp: float
    open_idx: int
    horizon_idx: int


def _validate_data(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    return ensure_backtest_ready(df, price_col=price_col).copy()


def _compute_signal_contract(
    df: pd.DataFrame, settings: BacktestSettings, price_col: str
) -> TechnicalSignal:
    """Return :class:`TechnicalSignal` for the last row of ``df``."""

    # ensure new contract API
    if hasattr(settings.strategy, "compat_int"):
        settings.strategy.compat_int = False

    sig = compute_signal(df, settings, price_col=price_col)
    if isinstance(sig, pd.Series):
        last = int(sig.iloc[-1]) if len(sig) else 0
        action = "BUY" if last > 0 else "SELL" if last < 0 else "KEEP"
        entry = float(df[price_col].iloc[-1]) if action != "KEEP" else 0.0
        return TechnicalSignal(
            timeframe=settings.timeframe,
            action=action,
            entry=entry,
        )
    return sig


def _fuse_with_time(
    tech_signal: int,
    ts: pd.Timestamp,
    price: float,
    time_model: TimeOnlyModel | None,
    min_conf: float,
    ai_decision: int | tuple[int, float] | None = None,
    *,
    return_reason: bool = False,
    return_weight: bool = False,
) -> int | tuple:
    """Fuse technical, time and optional AI signals into one decision."""

    votes: dict[str, tuple[int, float]] = {
        "tech": (_to_sign(tech_signal), 1.0),
        "time": (0, 0.0),
        "ai": (0, 0.0),
    }

    if time_model:
        tm_res = time_model.decide(ts)
        if isinstance(tm_res, dict):
            tm_decision = tm_res.get("decision")
            tm_weight = tm_res.get("confidence", 1.0)
        elif isinstance(tm_res, tuple):
            tm_decision, tm_weight = tm_res
        else:  # pragma: no cover - simple fallback
            tm_decision, tm_weight = tm_res, 1.0
        if tm_decision in {"WAIT", "HOLD"}:
            reason = "time_model_hold" if tm_decision == "HOLD" else "time_model_wait"
            if return_reason and return_weight:
                return 0, 0.0, reason
            if return_reason:
                return 0, reason
            if return_weight:
                return 0, 0.0
            return 0
        votes["time"] = (_to_sign(1 if tm_decision == "BUY" else -1), float(tm_weight))

    if ai_decision is not None:
        if isinstance(ai_decision, tuple):
            ai_sig, ai_weight = ai_decision
        else:
            ai_sig, ai_weight = ai_decision, 1.0
        votes["ai"] = (_to_sign(ai_sig), float(ai_weight))

    pos_total = sum(w for s, w in votes.values() if s > 0)
    neg_total = sum(w for s, w in votes.values() if s < 0)
    if max(pos_total, neg_total) < max(min_conf, 1.0):
        if return_reason and return_weight:
            return 0, 0.0, "not_enough_confluence"
        if return_reason:
            return 0, "not_enough_confluence"
        if return_weight:
            return 0, 0.0
        return 0

    if pos_total > neg_total:
        res_dec = 1
        weights = [w for s, w in votes.values() if s > 0]
    elif neg_total > pos_total:
        res_dec = -1
        weights = [w for s, w in votes.values() if s < 0]
    else:
        res_dec = 0
        weights = []

    final_weight = min(weights) if weights else 0.0

    if return_reason and return_weight:
        return res_dec, final_weight, None
    if return_weight:
        return res_dec, final_weight
    if return_reason:
        return res_dec, None
    return res_dec


def bootstrap_position(
    df: pd.DataFrame,
    sig: pd.Series,
    rm: RiskManager,
    tb: TradeBook,
    settings: BacktestSettings,
    price_col: str,
    atr_multiple: float,
) -> float:
    """Attempt to open a position on the first bar for extreme EMA params."""

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


def on_bar_close(
    df: pd.DataFrame,
    idx: int,
    settings: BacktestSettings,
    registry: SetupRegistry,
    price_col: str,
) -> None:
    """Compute signal for bar ``idx`` and arm setups for the next bar."""

    sig = _compute_signal_contract(df.iloc[: idx + 1], settings, price_col)
    if sig.action not in {"BUY", "SELL"}:
        return

    tf_minutes = _TF_MINUTES.get(settings.timeframe, 0)
    horizon_bars = 0
    if tf_minutes > 0 and sig.horizon_minutes:
        horizon_bars = int(np.ceil(sig.horizon_minutes / tf_minutes))

    sig.meta["armed_index"] = idx
    sig.meta["horizon_bars"] = horizon_bars
    registry.arm(settings.timeframe, idx, sig)


def _apply_trailing(position: Position, high: float, atr_val: float, atr_mult: float) -> None:
    if atr_val <= 0:
        return
    new_sl = max(position.sl, high - atr_mult * atr_val)
    position.sl = float(new_sl)


def _check_exit(
    position: Position,
    high: float,
    low: float,
    policy: TPslPolicy,
) -> float | None:
    """Return exit price if TP or SL is hit according to ``policy``."""

    if policy == TPslPolicy.TP:
        if high >= position.tp >= 0:
            return position.tp
        if low <= position.sl <= position.tp:
            return position.sl
    else:
        if low <= position.sl <= position.tp:
            return position.sl
        if high >= position.tp >= 0:
            return position.tp
    return None


def on_bar_open(
    idx: int,
    row: pd.Series,
    settings: BacktestSettings,
    registry: SetupRegistry,
    rm: RiskManager,
    tb: TradeBook,
    position: Position | None,
    policy: TPslPolicy,
    time_model: TimeOnlyModel | None,
    atr_multiple: float,
    price_col: str,
) -> Position | None:
    """Process open positions and trigger setups at the start of a bar."""

    high = float(row["high"])
    low = float(row["low"])
    atr_val = float(row.get("atr", 0.0))

    # --- manage existing position ---------------------------------------
    if position is not None:
        _apply_trailing(position, high, atr_val, atr_multiple)
        exit_price = _check_exit(position, high, low, policy)
        if exit_price is not None:
            rm.sell(exit_price, position.qty)
            tb.add(row.name, exit_price, position.qty, "SELL")
            position = None
        else:
            horizon_ok = idx < position.horizon_idx
            max_bars = getattr(settings, "max_bars_open", 0)
            if not horizon_ok or (max_bars > 0 and idx - position.open_idx >= max_bars):
                close_price = float(row[price_col])
                rm.sell(close_price, position.qty)
                tb.add(row.name, close_price, position.qty, "SELL")
                position = None

    # --- try opening a new position -------------------------------------
    if position is None:
        trig = registry.check(settings.timeframe, idx, high, low)
        if trig is not None:
            armed_idx = trig.meta.get("armed_index", idx - 1)
            horizon = trig.meta.get("horizon_bars", 0)
            if horizon and idx - armed_idx > horizon:
                trig = None
        if trig is not None and trig.action == "BUY":
            if time_model is not None:
                tm_res = time_model.decide(row.name)
                decision = tm_res.get("decision") if isinstance(tm_res, dict) else tm_res[0]
                if decision in {"WAIT", "HOLD"}:
                    trig = None
            if trig is not None:
                open_price = float(row["open"])
                entry = float(trig.entry)
                fill = open_price if open_price > entry else entry
                qty = rm.position_size(
                    price=fill,
                    atr=atr_val,
                    atr_multiple=atr_multiple,
                )
                if qty > 0.0:
                    rm.buy(fill, qty)
                    tb.add(row.name, fill, qty, "BUY")
                    horizon_idx = idx + trig.meta.get("horizon_bars", 0)
                    if horizon_idx == idx:
                        horizon_idx = idx + 10**9
                    position = Position(
                        qty=qty,
                        entry=fill,
                        sl=float(trig.sl),
                        tp=float(trig.tp),
                        open_idx=idx,
                        horizon_idx=horizon_idx,
                    )
    return position


def _compute_metrics(equity_curve: list[float]) -> tuple[pd.Series, float]:
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
    df = _validate_data(df, price_col=price_col)

    ap = int(atr_period or settings.atr_period)
    am = float(atr_multiple or settings.atr_multiple)
    df["atr"] = atr(df["high"], df["low"], df["close"], ap)

    tb = TradeBook()
    rm = risk or RiskManager(**settings.risk.model_dump())
    registry = SetupRegistry(getattr(settings, "setup_ttl_bars", 1))

    time_model: TimeOnlyModel | None = None
    if settings.time.model.enabled and settings.time.model.path:
        time_model = TimeOnlyModel.load(settings.time.model.path)

    policy = TPslPolicy(getattr(settings, "tp_sl_priority", "tp"))

    position: Position | None = None

    for idx in range(len(df)):
        row = df.iloc[idx]
        position = on_bar_open(
            idx,
            row,
            settings,
            registry,
            rm,
            tb,
            position,
            policy,
            time_model,
            am,
            price_col,
        )
        on_bar_close(df, idx, settings, registry, price_col)

        equity_mtm = rm.equity + (position.qty if position else 0.0) * float(row[price_col])
        rm.record_mark_to_market(equity_mtm)
        if rm.exceeded_max_dd():
            log.warning("max_dd_exceeded", time=str(df.index[idx]), equity=equity_mtm)
            break

    eq, max_dd = _compute_metrics(rm.equity_curve)
    return BacktestResult(equity_curve=eq, max_dd=max_dd, trades=tb)


__all__ = [
    "BacktestResult",
    "TPslPolicy",
    "run_backtest",
    "on_bar_open",
    "on_bar_close",
    "_compute_signal_contract",
]
