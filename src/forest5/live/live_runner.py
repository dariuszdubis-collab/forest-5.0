from __future__ import annotations

import json
import os
import time

import pandas as pd

from pathlib import Path

from ..config_live import LiveSettings
from ..decision import DecisionAgent, DecisionConfig
from ..time_only import TimeOnlyModel
from ..signals.factory import compute_signal
from ..signals.candles import candles_signal
from ..signals.combine import confirm_with_candles
from ..utils.timeframes import _TF_MINUTES
from ..utils.log import setup_logger
from ..utils.debugger import DebugLogger
from .router import OrderRouter
from .risk_guard import should_halt_for_drawdown
from ..utils.fs_watcher import wait_for_mtime


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


def _append_bar_and_signal(df: pd.DataFrame, bar: dict, settings: LiveSettings) -> int:
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

    # Only the EMA cross strategy is used in tests; fall back to the original
    # implementation for anything else.
    if getattr(settings.strategy, "name", "ema_cross") != "ema_cross":
        return int(compute_signal(df, settings, "close").iloc[-1])

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
        from .router import PaperBroker

        broker = PaperBroker()
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
        ),
    )

    context_text = ""
    if settings.ai.context_file:
        context_text = _read_context(settings.ai.context_file, 32_768)

    tf = settings.strategy.timeframe
    bar_sec = _TF_MINUTES[tf] * 60

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
            remaining = timeout - (time.time() - last_candle_ts)
            if remaining <= 0:
                log.info("idle_timeout_reached")
                break

            mtime = wait_for_mtime(tick_file, last_mtime, remaining)
            if mtime is None:
                continue
            last_mtime = mtime
            try:
                tick = json.loads(_read_context(tick_file, 4096))
            except json.JSONDecodeError:  # pragma: no cover - defensive
                log.exception("invalid tick data")
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

            if bar_start == current_bar["start"]:
                current_bar["high"] = max(current_bar["high"], price)
                current_bar["low"] = min(current_bar["low"], price)
                current_bar["close"] = price
            else:
                idx = pd.to_datetime(current_bar["start"], unit="s")
                sig = _append_bar_and_signal(df, current_bar, settings)
                log.info("candle_closed", **current_bar)
                last_candle_ts = time.time()

                cur_eq = broker.equity()
                if cur_eq is not None and should_halt_for_drawdown(
                    start_equity, cur_eq, settings.risk.max_drawdown
                ):
                    dd = (start_equity - cur_eq) / start_equity
                    log.error("max_drawdown_reached", drawdown_pct=dd * 100)
                    break

                if (
                    idx.weekday() in settings.time.blocked_weekdays
                    or idx.hour in settings.time.blocked_hours
                ):
                    log.info("time_blocked", time=str(idx))
                    if debug:
                        debug.log("skip_candle", time=str(idx), reason="time_block")
                else:
                    decision, votes, reason = agent.decide(
                        idx,
                        sig,
                        current_bar["close"],
                        settings.broker.symbol,
                        context_text,
                    )
                    log.info(
                        "decision",
                        timestamp=time.time(),
                        symbol=settings.broker.symbol,
                        action="decision",
                        side=decision,
                        qty=(settings.broker.volume if decision in ("BUY", "SELL") else 0),
                        price=current_bar["close"],
                        latency_ms=0.0,
                        error=None,
                        context={"votes": votes, "reason": reason},
                    )
                    if debug:
                        debug.log(
                            "decision",
                            time=str(idx),
                            decision=decision,
                            votes=votes,
                            reason=reason,
                        )
                    if decision in ("BUY", "SELL"):
                        start_ts = time.time()
                        res = broker.market_order(decision, settings.broker.volume, price)
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

                current_bar = {
                    "start": bar_start,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                }

                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
    except KeyboardInterrupt:
        log.info("keyboard_interrupt")
    finally:
        pos = broker.position_qty()
        if pos > 0:
            broker.market_order("SELL", pos, last_price)
        broker.close()
        if debug:
            debug.close()
