from __future__ import annotations

import json
import time

import pandas as pd

from pathlib import Path

from ..config_live import LiveSettings
from ..decision import DecisionAgent, DecisionConfig
from ..time_only import TimeOnlyModel
from ..signals.factory import compute_signal
from ..utils.timeframes import _TF_MINUTES
from .risk_guard import should_halt_for_drawdown
from ..utils.log import log
from .router import OrderRouter


def _read_context(path: str | Path, max_bytes: int) -> str:
    """Read up to ``max_bytes`` from ``path`` as UTF-8 text.

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


def run_live(
    settings: LiveSettings,
    *,
    max_steps: int | None = None,
    timeout: float = 2.0,
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

    time_model: TimeOnlyModel | None = None
    if settings.time.model.enabled and settings.time.model.path:
        try:
            time_model = TimeOnlyModel.load(settings.time.model.path)
        except (OSError, json.JSONDecodeError, KeyError):  # pragma: no cover - defensive
            log.exception("failed to load time model")

    agent = DecisionAgent(
        router=broker,
        config=DecisionConfig(
            use_ai=settings.ai.enabled,
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
    risk_halt = False

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

                    if bar_start == current_bar["start"]:
                        current_bar["high"] = max(current_bar["high"], price)
                        current_bar["low"] = min(current_bar["low"], price)
                        current_bar["close"] = price
                    else:
                        idx = pd.to_datetime(current_bar["start"], unit="s")
                        df.loc[idx] = [
                            current_bar["open"],
                            current_bar["high"],
                            current_bar["low"],
                            current_bar["close"],
                        ]
                        log.info("candle_closed", **current_bar)
                        last_candle_ts = time.time()

                        cur_eq = broker.equity()
                        if cur_eq is not None and should_halt_for_drawdown(
                            start_equity,
                            cur_eq,
                            settings.risk.max_drawdown,
                        ):
                            dd = (start_equity - cur_eq) / start_equity
                            if settings.risk.on_drawdown.action == "halt":
                                log.error("risk_guard_halt", drawdown_pct=dd * 100)
                                break
                            elif settings.risk.on_drawdown.action == "soft_wait":
                                risk_halt = True
                                log.info("risk_guard_halt", drawdown_pct=dd * 100)

                        if (
                            idx.weekday() in settings.time.blocked_weekdays
                            or idx.hour in settings.time.blocked_hours
                        ):
                            log.info("time_blocked", time=str(idx))
                        else:
                            sig = int(compute_signal(df, settings, "close").iloc[-1])
                            decision, votes, reason = agent.decide(
                                idx,
                                sig,
                                current_bar["close"],
                                settings.broker.symbol,
                                context_text,
                            )
                            log.info(
                                "decision",
                                time=str(idx),
                                symbol=settings.broker.symbol,
                                decision=decision,
                                votes=votes,
                                reason=reason,
                            )
                            if decision in ("BUY", "SELL"):
                                if risk_halt:
                                    log.info("risk_guard_active", decision=decision)
                                else:
                                    broker.market_order(decision, settings.broker.volume, price)

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
            time.sleep(0.25)
    except KeyboardInterrupt:
        log.info("keyboard_interrupt")
    finally:
        pos = broker.position_qty()
        if pos > 0:
            broker.market_order("SELL", pos, last_price)
        broker.close()
