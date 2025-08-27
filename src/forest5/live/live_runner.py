from __future__ import annotations

import json
import os
import time

import pandas as pd

from pathlib import Path

from ..config_live import LiveSettings
from ..decision import DecisionAgent, DecisionConfig, DecisionResult
from ..time_only import TimeOnlyModel
from ..signals.candles import candles_signal
from ..signals.combine import confirm_with_candles
from ..signals.compat import compute_signal_compat
from ..signals import SetupRegistry, compute_signal
from ..signals.contract import TechnicalSignal
from ..utils.timeframes import _TF_MINUTES
from ..utils.log import (
    setup_logger,
    TelemetryContext,
    new_id,
    log_event,
    E_SETUP_ARM,
    E_SETUP_TRIGGER,
)
from ..utils.debugger import DebugLogger
import forest5.live.router as router
from .risk_guard import should_halt_for_drawdown

OrderRouter = router.OrderRouter


log = setup_logger()


def _read_context(path: str | Path, max_bytes: int = 16_384) -> str:
    """Read up to ``max_bytes`` bytes from ``path`` as UTF-8 text.

    Bytes that cannot be decoded are ignored. Any I/O errors are logged and an
    empty string is returned.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_bytes)
        return data.decode("utf-8", errors="ignore")
    except OSError:  # pragma: no cover - defensive
        log.exception("failed to read context file", path=str(path))
        return ""


def append_bar_and_signal(
    df: pd.DataFrame,
    bar: dict,
    settings: LiveSettings,
    *,
    setup_registry: SetupRegistry | None = None,
    ctx: TelemetryContext | None = None,
) -> int:
    """Append ``bar`` to ``df`` and return only the latest signal.

    This updates EMA values incrementally so indicator calculations scale
    with the number of new bars instead of the full history.
    """
    idx = pd.to_datetime(bar["start"], unit="s")
    df.loc[idx, ["open", "high", "low", "close"]] = [
        bar["open"],
        bar["high"],
        bar["low"],
        bar["close"],
    ]

    name = getattr(settings.strategy, "name", "ema_cross")
    if name == "h1_ema_rsi_atr":
        # For the H1 strategy we work with rich contract objects.  Compute the
        # signal directly without going through the compatibility shim.
        contract = compute_signal(df, settings, ctx=ctx)
        if (
            isinstance(contract, TechnicalSignal)
            and contract.action in ("BUY", "SELL")
            and setup_registry is not None
        ):
            signal = TechnicalSignal(
                timeframe=contract.timeframe,
                action=contract.action,
                entry=contract.entry,
                sl=contract.sl,
                tp=contract.tp,
                horizon_minutes=contract.horizon_minutes,
                technical_score=contract.technical_score,
                confidence_tech=contract.confidence_tech,
                drivers=contract.drivers,
                meta=contract.meta,
            )
            key = getattr(settings.broker, "symbol", "")
            setup_registry.arm(
                key,
                len(df),
                signal,
                bar_time=idx,
                ttl_minutes=getattr(settings.strategy, "setup_ttl_minutes", None),
                ctx=ctx,
            )
            log_event(
                E_SETUP_ARM,
                ctx=ctx,
                key=key,
                index=len(df),
                action=signal.action,
                entry=float(signal.entry),
                sl=float(signal.sl),
                tp=float(signal.tp),
            )
        return 0

    if name != "ema_cross":
        return int(compute_signal_compat(df, settings, "close").iloc[-1])

    close = float(bar["close"])
    fast = settings.strategy.fast
    slow = settings.strategy.slow
    alpha_fast = 2 / (fast + 1)
    alpha_slow = 2 / (slow + 1)

    if "ema_fast" not in df.columns:
        df["ema_fast"] = pd.Series(dtype=float)
        df["ema_slow"] = pd.Series(dtype=float)

    if len(df) > 1:
        prev_fast = float(df["ema_fast"].iloc[-2])
        prev_slow = float(df["ema_slow"].iloc[-2])
    else:
        prev_fast = prev_slow = close

    ema_fast = close * alpha_fast + prev_fast * (1 - alpha_fast)
    ema_slow = close * alpha_slow + prev_slow * (1 - alpha_slow)
    df.at[idx, "ema_fast"] = ema_fast
    df.at[idx, "ema_slow"] = ema_slow

    sig = 0
    if len(df) > 1:
        prev_dir = 1 if prev_fast > prev_slow else -1
        direction = 1 if ema_fast > ema_slow else -1
        if direction != prev_dir:
            sig = direction

    candle = candles_signal(df.iloc[-2:]).iloc[-1] if len(df) > 1 else 0
    if sig != 0 or candle != 0:
        idx_ser = pd.Series([sig], index=[idx])
        candle_ser = pd.Series([candle], index=[idx])
        sig = int(confirm_with_candles(idx_ser, candle_ser).iloc[-1])
    return sig


# Backward compatibility
_append_bar_and_signal = append_bar_and_signal


def run_live(
    settings: LiveSettings,
    *,
    max_steps: int | None = None,
    timeout: float = 2.0,
    debug_dir: Path | None = None,
) -> None:
    """Run the live trading loop.

    The loop processes incoming ticks, closes candles, and makes trading
    decisions. It will exit early if no new candles are processed within
    ``timeout`` seconds.

    Parameters
    ----------
    settings:
        Configuration for the live run.
    max_steps:
        Optional limit on number of candles to process.
    timeout:
        Seconds to wait for a new candle before stopping.
    """
    btype = settings.broker.type.lower()
    broker: OrderRouter
    if btype == "mt4":
        try:
            from .mt4_broker import MT4Broker
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError("MT4Broker import failed") from exc
        broker = MT4Broker(settings.broker.bridge_dir, symbol=settings.broker.symbol)
        tick_dir = broker.ticks_dir
    elif btype == "paper":
        broker = router.PaperBroker()
        if settings.broker.bridge_dir is None:
            raise ValueError("bridge_dir required for paper broker")
        tick_dir = Path(settings.broker.bridge_dir) / "ticks"
        broker.ticks_dir = tick_dir  # type: ignore[attr-defined]
    else:
        raise ValueError(f"unsupported broker type: {settings.broker.type}")

    broker.connect()
    start_equity = broker.equity() or 0.0
    debug: DebugLogger | None = None
    if debug_dir:
        session_dir = Path(debug_dir) / f"session_{int(time.time())}"
        debug = DebugLogger(session_dir)

    time_model: TimeOnlyModel | None = None
    tm_path = settings.time.model.path
    if settings.time.model.enabled and tm_path and Path(tm_path).exists():
        try:
            time_model = TimeOnlyModel.load(tm_path)
        except (OSError, json.JSONDecodeError, KeyError):  # pragma: no cover - defensive
            log.exception("failed to load time model")
    else:
        log.info("time_model_missing", path=tm_path)

    use_ai = settings.ai.enabled
    ctx_file = settings.ai.context_file or ""
    if use_ai and ctx_file and not Path(ctx_file).exists():
        if getattr(settings.ai, "require_context", False):
            log.warning("ai_context_missing", path=ctx_file)
            raise FileNotFoundError(ctx_file)
        log.warning("ai_context_missing_warn", path=ctx_file)
        use_ai = False
    if use_ai and not os.getenv("OPENAI_API_KEY"):
        log.info("ai_disabled_no_api_key")
        use_ai = False

    agent = DecisionAgent(
        router=broker,
        config=DecisionConfig(
            use_ai=use_ai,
            ai_model=settings.ai.model,
            ai_max_tokens=settings.ai.max_tokens,
            time_model=time_model,
            min_confluence=settings.decision.min_confluence,
            tie_epsilon=settings.decision.tie_epsilon,
            weights=settings.decision.weights,
            tech=settings.decision.tech,
        ),
    )

    context_text = ""
    if use_ai and ctx_file:
        context_text = _read_context(ctx_file, 32_768)

    tf = settings.strategy.timeframe
    bar_sec = _TF_MINUTES[tf] * 60

    run_id = new_id("run")
    registry = SetupRegistry()
    setup_registry = registry

    def mk_ctx(setup_id: str | None = None) -> TelemetryContext:
        return TelemetryContext(
            run_id=run_id,
            symbol=settings.broker.symbol,
            timeframe=tf,
            setup_id=setup_id,
        )

    tick_file = tick_dir / "tick.json"
    last_mtime = 0.0

    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    current_bar: dict | None = None
    last_price: float | None = None
    steps = 0
    # Track when the last candle was processed to detect idle periods
    last_candle_ts = time.time()

    try:
        while True:
            if time.time() - last_candle_ts > timeout:
                log.info("idle_timeout_reached")
                break

            if tick_file.exists():
                mtime = tick_file.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    try:
                        tick = json.loads(_read_context(tick_file, 4096))
                    except json.JSONDecodeError:  # pragma: no cover - defensive
                        log.exception("invalid tick data")
                        time.sleep(0.25)
                        continue

                    ts = float(tick.get("time", time.time()))
                    price = float(tick.get("bid") or tick.get("price") or tick.get("ask"))
                    last_price = price
                    log.info("tick", **tick)

                    bar_start = int(ts // bar_sec) * bar_sec
                    if current_bar is None:
                        current_bar = {
                            "start": bar_start,
                            "open": price,
                            "high": price,
                            "low": price,
                            "close": price,
                        }
                        continue

                    bar_closed = False
                    if bar_start == current_bar["start"]:
                        current_bar["high"] = max(current_bar["high"], price)
                        current_bar["low"] = min(current_bar["low"], price)
                        current_bar["close"] = price
                    else:
                        idx = pd.to_datetime(current_bar["start"], unit="s")
                        setup_registry = registry
                        ctx = TelemetryContext(
                            run_id=run_id,
                            symbol=settings.broker.symbol,
                            timeframe=tf,
                        )
                        try:
                            sig = append_bar_and_signal(
                                df,
                                current_bar,
                                settings,
                                setup_registry=setup_registry,
                                ctx=ctx,
                            )
                        except TypeError:
                            sig = append_bar_and_signal(df, current_bar, settings)
                        log.info("candle_closed", **current_bar)
                        last_candle_ts = time.time()

                        cur_eq = broker.equity()
                        if cur_eq is not None and should_halt_for_drawdown(
                            start_equity, cur_eq, settings.risk.max_drawdown
                        ):
                            dd = (start_equity - cur_eq) / start_equity
                            log.error("max_drawdown_reached", drawdown_pct=dd * 100)
                            break

                        if sig != 0:
                            action = "BUY" if sig > 0 else "SELL"
                            setup_id = new_id("setup")
                            signal = TechnicalSignal(
                                timeframe=tf,
                                action=action,
                                entry=current_bar["close"],
                                horizon_minutes=_TF_MINUTES[tf],
                                technical_score=1.0 if action == "BUY" else -1.0,
                                confidence_tech=1.0,
                            )
                            ctx_setup = TelemetryContext(
                                run_id=run_id,
                                symbol=settings.broker.symbol,
                                timeframe=tf,
                                setup_id=setup_id,
                            )
                            registry.arm(
                                setup_id,
                                len(df),
                                signal,
                                bar_time=pd.to_datetime(bar_start, unit="s"),
                                ctx=ctx_setup,
                            )

                        current_bar = {
                            "start": bar_start,
                            "open": price,
                            "high": price,
                            "low": price,
                            "close": price,
                        }
                        bar_closed = True

                    triggered = None
                    if bar_closed:
                        triggered = setup_registry.check(
                            index=len(df) - 1,
                            price=price,
                            now=pd.to_datetime(ts, unit="s"),
                            ctx=mk_ctx(),
                        )
                    if triggered:
                        slippage = (
                            price - float(triggered.entry)
                            if triggered.action.upper() == "BUY"
                            else float(triggered.entry) - price
                        )
                        log_event(
                            E_SETUP_TRIGGER,
                            ctx=mk_ctx(triggered.setup_id),
                            trigger_price=price,
                            fill_price=price,
                            slippage=slippage,
                            setup_id=triggered.setup_id,
                            action=triggered.action,
                            entry=float(triggered.entry),
                            sl=float(triggered.sl),
                            tp=float(triggered.tp),
                        )
                        idx = pd.to_datetime(ts, unit="s")
                        dec: DecisionResult = agent.decide(
                            idx,
                            triggered,
                            price,
                            settings.broker.symbol,
                            context_text,
                        )
                        decision = dec.decision
                        weight = dec.weight_sum
                        votes = dec.votes
                        reason = dec.reason
                        log.info(
                            "decision",
                            timestamp=time.time(),
                            symbol=settings.broker.symbol,
                            action="decision",
                            side=decision,
                            qty=(
                                settings.broker.volume * weight
                                if decision in ("BUY", "SELL")
                                else 0
                            ),
                            price=price,
                            latency_ms=0.0,
                            error=None,
                            context={"votes": votes, "reason": reason, "weight": weight},
                        )
                        if debug:
                            debug.log(
                                "decision",
                                time=str(idx),
                                decision=decision,
                                votes=votes,
                                reason=reason,
                                weight=float(weight),
                            )
                        if decision in ("BUY", "SELL") and weight > 0:
                            start_ts = time.time()
                            res = broker.market_order(
                                decision,
                                settings.broker.volume * weight,
                                price,
                                entry=triggered.entry,
                                sl=triggered.sl,
                                tp=triggered.tp,
                            )
                            latency = (time.time() - start_ts) * 1000.0
                            log.info(
                                "order_result",
                                timestamp=time.time(),
                                symbol=settings.broker.symbol,
                                action="market_order",
                                side=decision,
                                qty=res.filled_qty,
                                price=res.avg_price,
                                latency_ms=latency,
                                error=res.error,
                                context={"status": res.status, "id": res.id},
                            )
                            if debug:
                                debug.log(
                                    "order",
                                    time=str(idx),
                                    side=decision,
                                    qty=res.filled_qty,
                                    price=res.avg_price,
                                    status=res.status,
                                    latency_ms=latency,
                                )
                    elif not getattr(setup_registry, "_setups", {}) and bar_closed:
                        idx = pd.to_datetime(ts, unit="s")
                        dec = agent.decide(
                            idx,
                            0,
                            price,
                            settings.broker.symbol,
                            context_text,
                        )
                        decision = dec.decision
                        weight = dec.weight_sum
                        votes = dec.votes
                        reason = dec.reason
                        log.info(
                            "decision",
                            timestamp=time.time(),
                            symbol=settings.broker.symbol,
                            action="decision",
                            side=decision,
                            qty=(
                                settings.broker.volume * weight
                                if decision in ("BUY", "SELL")
                                else 0
                            ),
                            price=price,
                            latency_ms=0.0,
                            error=None,
                            context={"votes": votes, "reason": reason, "weight": weight},
                        )
                        if debug:
                            debug.log(
                                "decision",
                                time=str(idx),
                                decision=decision,
                                votes=votes,
                                reason=reason,
                                weight=float(weight),
                            )
                        if decision in ("BUY", "SELL") and weight > 0:
                            start_ts = time.time()
                            res = broker.market_order(
                                decision,
                                settings.broker.volume * weight,
                                price,
                            )
                            latency = (time.time() - start_ts) * 1000.0
                            log.info(
                                "order_result",
                                timestamp=time.time(),
                                symbol=settings.broker.symbol,
                                action="market_order",
                                side=decision,
                                qty=res.filled_qty,
                                price=res.avg_price,
                                latency_ms=latency,
                                error=res.error,
                                context={"status": res.status, "id": res.id},
                            )
                            if debug:
                                debug.log(
                                    "order",
                                    time=str(idx),
                                    side=decision,
                                    qty=res.filled_qty,
                                    price=res.avg_price,
                                    status=res.status,
                                    latency_ms=latency,
                                )

                    if bar_closed:
                        steps += 1
                        if max_steps is not None and steps >= max_steps:
                            break
            time.sleep(0.25)
    except KeyboardInterrupt:
        log.info("keyboard_interrupt")
    finally:
        pos = broker.position_qty()
        if pos > 0:
            broker.market_order("SELL", pos, last_price)
        broker.close()
        if debug:
            debug.close()
