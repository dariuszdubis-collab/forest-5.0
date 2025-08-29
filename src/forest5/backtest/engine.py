from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from ..config import BacktestSettings
from ..utils.debugger import DebugLogger, TraceCollector
from ..core.indicators import atr, ema, rsi
from ..utils.log import (
    E_ORDER_ACK,
    E_ORDER_FILLED,
    E_ORDER_REJECTED,
    E_ORDER_SUBMITTED,
    TelemetryContext,
    log_event,
    new_id,
    setup_logger,
)
from ..utils.validate import ensure_backtest_ready
from forest5.signals.factory import compute_signal
from ..signals.contract import TechnicalSignal
from ..signals.setups import SetupRegistry, TriggeredSignal
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


@dataclass
class TPslPolicy:
    """Policy defining priority between take-profit and stop-loss hits."""

    priority: str = "SL_FIRST"


def _validate_data(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Zapewnia poprawność danych wejściowych do backtestu."""
    out = ensure_backtest_ready(df, price_col=price_col)
    float_cols = out.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        out[float_cols] = out[float_cols].astype("float32")
    return out


def _generate_signal(
    df: pd.DataFrame,
    settings: BacktestSettings,
    price_col: str,
    *,
    collector: TraceCollector | None = None,
) -> pd.Series:
    """Generuje serię sygnałów tradingowych."""
    name = getattr(settings.strategy, "name", "ema_cross")
    use_rsi = getattr(settings.strategy, "use_rsi", False)
    if name in {"ema_rsi", "ema-cross+rsi"}:
        use_rsi = True

    if name == "h1_ema_rsi_atr":
        # Strategy returns a contract for the latest bar only. Build a signal
        # series by iterating through the DataFrame and converting each contract
        # to ``-1/0/1``.
        from forest5.signals.h1_ema_rsi_atr import compute_primary_signal_h1
        from forest5.signals.compat import contract_to_int

        registry = SetupRegistry()
        params = getattr(settings.strategy, "params", settings.strategy)
        vals: list[int] = []
        for i in range(len(df)):
            contract = compute_primary_signal_h1(
                df.iloc[: i + 1],
                params=params,
                registry=registry,
                collector=collector,
            )
            vals.append(int(contract_to_int(contract)))
        sig = pd.Series(vals, index=df.index, dtype=int)
    else:
        run_ctx = TelemetryContext(
            run_id=new_id("run"),
            symbol=settings.symbol,
            timeframe=settings.tf.name if hasattr(settings, "tf") else "H1",
            strategy=settings.name if hasattr(settings, "name") else "unknown",
        )
        try:
            res = compute_signal(
                df,
                settings,
                price_col=price_col,
                compat_int=False,
                ctx=run_ctx,
            )
        except TypeError:
            res = compute_signal(
                df,
                settings,
                price_col=price_col,
                compat_int=False,
            )
        if isinstance(res, TechnicalSignal):
            from forest5.signals.compat import contract_to_int

            val = int(contract_to_int(res))
            sig = pd.Series([val] * len(df), index=df.index, dtype=int)
        else:
            sig = res.astype(int)

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
                tb.add(df.index[0], p0, qty0, "BUY", entry=p0)
                position += qty0
    return position


class BacktestEngine:
    """Event driven backtesting engine with setup management."""

    def __init__(
        self, df: pd.DataFrame, settings: BacktestSettings, price_col: str = "close"
    ) -> None:
        self.df = df
        self.settings = settings
        self.price_col = price_col

        # Core components
        self.tp_sl_policy = TPslPolicy(priority=getattr(settings, "tp_sl_priority", "SL_FIRST"))
        ttl = getattr(
            settings,
            "setup_ttl_bars",
            getattr(getattr(settings, "strategy", object()), "setup_ttl_bars", 1),
        )
        self.setups = SetupRegistry(ttl_bars=int(ttl))
        ttl_minutes = getattr(
            settings,
            "setup_ttl_minutes",
            getattr(getattr(settings, "strategy", object()), "setup_ttl_minutes", None),
        )
        self._setup_ttl_minutes = int(ttl_minutes) if ttl_minutes is not None else None

        self.time_model: TimeOnlyModel | None = None
        if settings.time.model.enabled and settings.time.model.path:
            try:
                self.time_model = TimeOnlyModel.load(settings.time.model.path)
            except Exception:  # pragma: no cover - defensive
                self.time_model = None

        self.positions: list[dict] = []
        self.equity: float = 0.0
        self.equity_curve: list[float] = []
        self.run_id = new_id("run")
        self._ticket_seq = 0

    # ------------------------------------------------------------------
    # Signal generation and setup handling
    def _compute_signal_contract(self, up_to: int) -> TechnicalSignal:
        """Return full :class:`TechnicalSignal` for bar ``up_to``."""

        res = compute_signal(
            self.df.iloc[: up_to + 1],
            self.settings,
            price_col=self.price_col,
            compat_int=False,
        )
        if isinstance(res, TechnicalSignal):
            return res
        action_val = int(res.iloc[-1]) if len(res) else 0
        action = "BUY" if action_val > 0 else "SELL" if action_val < 0 else "KEEP"
        return TechnicalSignal(action=action)

    # ------------------------------------------------------------------
    def on_bar_close(self, index: int) -> None:
        """Handle bar close events.

        Updates open positions for the finished bar and arms new setups if the
        computed signal indicates so.
        """

        row = self.df.iloc[index]
        self._update_open_positions(index, row)

        contract = self._compute_signal_contract(index)
        if contract.action in {"BUY", "SELL"}:
            setup_id = str(index)
            if contract.meta:
                setup_id = str(contract.meta.get("id", setup_id))
                contract.meta = dict(contract.meta)
            else:
                contract.meta = {}
            contract.meta.setdefault("id", setup_id)
            ctx = TelemetryContext(
                run_id=self.run_id,
                symbol=self.settings.symbol,
                timeframe=self.settings.tf.name if hasattr(self.settings, "tf") else "H1",
                setup_id=setup_id,
            )
            bar_time = self.df.index[index]
            if not isinstance(bar_time, datetime):
                bar_time = pd.Timestamp(bar_time).to_pydatetime()
            self.setups.arm(
                setup_id,
                index,
                contract,
                bar_time=bar_time,
                ttl_minutes=self._setup_ttl_minutes,
                ctx=ctx,
            )

        # Mark-to-market after bar close
        close_price = float(row[self.price_col])
        mtm = 0.0
        for pos in self.positions:
            if pos["action"] == "BUY":
                mtm += close_price - pos["entry"]
            else:
                mtm += pos["entry"] - close_price
        self.equity_curve.append(self.equity + mtm)

    # ------------------------------------------------------------------
    def on_bar_open(self, index: int) -> None:
        """Trigger armed setups at the open of bar ``index``."""

        row = self.df.iloc[index]
        open_p = float(row["open"])
        now = self.df.index[index]
        if not isinstance(now, datetime):
            now = pd.Timestamp(now).to_pydatetime()

        while True:
            cand = self.setups.check(index=index, price=open_p, now=now)
            if cand is None:
                break

            # Gap fill: choose best fill price respecting direction
            entry = float(cand.entry)
            if cand.action == "BUY" and open_p > entry:
                entry = open_p
            elif cand.action == "SELL" and open_p < entry:
                entry = open_p

            # TimeOnlyModel check
            if self.time_model:
                tm_res = self.time_model.decide(self.df.index[index])
                decision = tm_res.get("decision") if isinstance(tm_res, dict) else tm_res
                if decision in {"WAIT", "HOLD"}:
                    continue  # blocked by time model

            self._open_position(cand, entry, index)

    # ------------------------------------------------------------------
    def _open_position(
        self, cand: TriggeredSignal | TechnicalSignal, entry: float, index: int
    ) -> None:
        setup_id = getattr(cand, "setup_id", getattr(cand, "id", ""))
        meta = dict(getattr(cand, "meta", {}) or {})
        qty = float(meta.get("qty", 1.0))
        client_id = new_id("cl")
        self._ticket_seq += 1
        ticket = self._ticket_seq
        ctx = TelemetryContext(
            run_id=self.run_id,
            symbol=self.settings.symbol,
            setup_id=setup_id,
            order_id=client_id,
        )
        if qty <= 0:
            log_event(
                E_ORDER_REJECTED,
                ctx,
                client_order_id=client_id,
                reason="qty_zero",
            )
            return
        log_event(
            E_ORDER_SUBMITTED,
            ctx,
            side=cand.action,
            volume=qty,
            price=entry,
            sl=float(cand.sl),
            tp=float(cand.tp),
            client_order_id=client_id,
        )
        log_event(E_ORDER_ACK, ctx, client_order_id=client_id, ticket=ticket)
        log_event(
            E_ORDER_FILLED,
            ctx,
            client_order_id=client_id,
            ticket=ticket,
            fill_price=entry,
            fill_qty=qty,
        )
        pos = {
            "id": setup_id,
            "action": cand.action,
            "entry": float(entry),
            # Preserve original SL/TP from the signal so risk logic doesn't
            # override them later.
            "sl": float(cand.sl),
            "tp": float(cand.tp),
            "orig_sl": float(cand.sl),
            "orig_tp": float(cand.tp),
            "open_index": index,
            "horizon": 0,
            "meta": meta,
            "ticket": ticket,
            "client_order_id": client_id,
            "qty": qty,
        }
        if "trailing_atr" in meta:
            pos["trailing_atr"] = float(meta["trailing_atr"])
        self.positions.append(pos)

    # ------------------------------------------------------------------
    def _close_position(self, pos: dict, price: float) -> None:
        side = "SELL" if pos["action"] == "BUY" else "BUY"
        qty = float(pos.get("qty", 1.0))
        client_id = new_id("cl")
        self._ticket_seq += 1
        ticket = self._ticket_seq
        ctx = TelemetryContext(
            run_id=self.run_id,
            symbol=self.settings.symbol,
            setup_id=str(pos.get("id", "")),
            order_id=client_id,
        )
        if qty <= 0:
            log_event(
                E_ORDER_REJECTED,
                ctx,
                client_order_id=client_id,
                reason="qty_zero",
            )
            return
        log_event(
            E_ORDER_SUBMITTED,
            ctx,
            side=side,
            volume=qty,
            price=price,
            client_order_id=client_id,
        )
        log_event(E_ORDER_ACK, ctx, client_order_id=client_id, ticket=ticket)
        log_event(
            E_ORDER_FILLED,
            ctx,
            client_order_id=client_id,
            ticket=ticket,
            fill_price=price,
            fill_qty=qty,
        )
        if pos["action"] == "BUY":
            self.equity += qty * (price - pos["entry"])
        else:
            self.equity += qty * (pos["entry"] - price)

    # ------------------------------------------------------------------
    def _update_open_positions(self, index: int, row: pd.Series) -> None:
        if not self.positions:
            return

        open_p = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row[self.price_col])
        atr_prev = (
            float(self.df.iloc[index - 1]["atr"])
            if "atr" in self.df.columns and index > 0
            else None
        )

        remaining: list[dict] = []
        for pos in self.positions:
            # Trailing stop update – only when original signal had no SL.
            trail = pos.get("trailing_atr")
            sl_orig = pos.get("orig_sl", pos.get("sl", 0.0))
            if trail is not None and atr_prev is not None and sl_orig <= 0.0:
                if pos["action"] == "BUY":
                    new_sl = close - trail * atr_prev
                    if pos["sl"] <= 0.0 or new_sl > pos["sl"]:
                        pos["sl"] = new_sl
                else:
                    new_sl = close + trail * atr_prev
                    if pos["sl"] <= 0.0 or new_sl < pos["sl"]:
                        pos["sl"] = new_sl

            bars_open = index - pos["open_index"]
            ttl = pos["meta"].get("ttl_bars") if isinstance(pos.get("meta"), dict) else None
            if ttl is not None and bars_open >= ttl:
                self._close_position(pos, close)
                continue

            horizon = pos.get("horizon")
            if horizon and bars_open >= horizon:
                self._close_position(pos, close)
                continue

            # Check for SL/TP hits, considering gaps at open first
            hit_tp = hit_sl = False
            fill_price = close
            if pos["action"] == "BUY":
                open_hit_sl = open_p <= pos["sl"]
                open_hit_tp = open_p >= pos["tp"]
                if open_hit_sl or open_hit_tp:
                    hit_sl, hit_tp = open_hit_sl, open_hit_tp
                    fill_price = open_p
                else:
                    high_hit = high >= pos["tp"]
                    low_hit = low <= pos["sl"]
                    hit_tp, hit_sl = high_hit, low_hit
                    fill_price = pos["tp"] if high_hit else pos["sl"] if low_hit else close
            else:  # SELL
                open_hit_sl = open_p >= pos["sl"]
                open_hit_tp = open_p <= pos["tp"]
                if open_hit_sl or open_hit_tp:
                    hit_sl, hit_tp = open_hit_sl, open_hit_tp
                    fill_price = open_p
                else:
                    high_hit = high >= pos["sl"]
                    low_hit = low <= pos["tp"]
                    hit_tp, hit_sl = low_hit, high_hit
                    fill_price = pos["tp"] if low_hit else pos["sl"] if high_hit else close

            if hit_tp and hit_sl:
                if self.tp_sl_policy.priority.upper() == "SL_FIRST":
                    hit_tp = False
                else:
                    hit_sl = False

            if hit_tp:
                self._close_position(pos, fill_price if fill_price != close else pos["tp"])
                continue
            if hit_sl:
                self._close_position(pos, fill_price if fill_price != close else pos["sl"])
                continue

            remaining.append(pos)

        self.positions = remaining


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
    """Fuse technical, time and optional AI signals into one decision.

    By default only the final decision ``-1/0/1`` is returned to preserve
    backwards compatibility. When ``return_weight`` is True, the confidence
    weight (0–1) is also returned. ``return_reason`` adds a textual reason.
    """

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
        elif isinstance(tm_res, tuple):  # backward compatibility
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
    collector: TraceCollector | None = None,
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

        fused, weight, fuse_reason = _fuse_with_time(
            this_sig,
            t,
            float(price),
            time_model,
            settings.time.fusion_min_confluence,
            None,
            return_reason=True,
            return_weight=True,
        )
        this_sig = fused
        if fuse_reason in {"time_model_hold", "time_model_wait"}:
            if debug:
                debug.log("skip_candle", time=str(t), reason=fuse_reason)
            if collector:
                collector.note("timeonly_wait", fuse_reason, at=pd.Timestamp(t), extras={})
        if fuse_reason == "not_enough_confluence" and debug:
            debug.log("signal_rejected", time=str(t), reason="no_confluence")

        if this_sig < 0 and position > 0.0:
            entry_price = getattr(rm, "_avg_price", 0.0)
            rm.sell(price, position)
            tb.add(t, entry_price, position, "SELL", price_close=float(price), entry=entry_price)
            if debug:
                debug.log("position_close", time=str(t), price=float(price), qty=float(position))
            position = 0.0

        if this_sig > 0 and position <= 0.0:
            qty = rm.position_size(price=price, atr=float(atr_val), atr_multiple=atr_multiple)
            qty *= float(weight)
            if qty > 0.0:
                rm.buy(price, qty)
                tb.add(t, price, qty, "BUY", entry=float(price))
                if debug:
                    debug.log(
                        "position_open",
                        time=str(t),
                        price=float(price),
                        qty=float(qty),
                        weight=float(weight),
                    )
                if collector:
                    collector.note(
                        "order_submit",
                        "ok",
                        at=pd.Timestamp(t),
                        extras={"price": float(price), "qty": float(qty)},
                    )
                position = qty
            else:
                if debug:
                    debug.log(
                        "signal_rejected",
                        time=str(t),
                        reason="qty_zero",
                        price=float(price),
                        atr=float(atr_val),
                    )
                if collector:
                    collector.note(
                        "order_reject",
                        "qty_zero",
                        at=pd.Timestamp(t),
                        extras={"price": float(price)},
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
    collector: TraceCollector | None = None,
) -> BacktestResult:
    # 1) walidacja danych
    df = _validate_data(df, price_col=price_col)

    # 2) generowanie sygnału
    sig = _generate_signal(df, settings, price_col=price_col, collector=collector)

    # 3) przygotowanie ATR do sizingu
    ap = int(atr_period or settings.atr_period)
    am = float(atr_multiple or settings.atr_multiple)
    df["atr"] = atr(df["high"], df["low"], df["close"], ap).astype("float32")

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
            collector,
        )
    finally:
        if debug:
            debug.close()

    # 7) metryki
    eq, max_dd = _compute_metrics(rm.equity_curve)
    return BacktestResult(equity_curve=eq, max_dd=max_dd, trades=tb)
