from __future__ import annotations

import json
import logging
import time

import pandas as pd

from pathlib import Path

from ..config_live import LiveSettings
from ..decision import DecisionAgent, DecisionConfig
from ..time_only import TimeOnlyModel
from ..signals.factory import compute_signal
from ..utils.timeframes import _TF_MINUTES

log = logging.getLogger(__name__)


def run_live(settings: LiveSettings, *, max_steps: int | None = None) -> None:
    btype = settings.broker.type.lower()
    if btype == "mt4":
        try:
            from .mt4_broker import MT4Broker
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("MT4Broker import failed") from exc
        broker = MT4Broker(settings.broker.bridge_dir, symbol=settings.broker.symbol)
    elif btype == "paper":
        from .router import PaperBroker

        broker = PaperBroker()
        if settings.broker.bridge_dir is None:
            raise ValueError("bridge_dir required for paper broker")
        broker.ticks_dir = Path(settings.broker.bridge_dir) / "ticks"  # type: ignore[attr-defined]
    else:
        raise ValueError(f"unsupported broker type: {settings.broker.type}")

    broker.connect()
    start_equity = broker.equity() or 0.0

    time_model = None
    if settings.time.model.enabled and settings.time.model.path:
        try:
            time_model = TimeOnlyModel.load(settings.time.model.path)
        except Exception:  # pragma: no cover - defensive
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
        try:
            context_text = Path(settings.ai.context_file).read_text(encoding="utf-8")
        except Exception:  # pragma: no cover - defensive
            log.exception("failed to read AI context file")

    tf = settings.strategy.timeframe
    bar_sec = _TF_MINUTES[tf] * 60

    tick_file = broker.ticks_dir / "tick.json"
    last_mtime = 0.0

    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    current_bar: dict | None = None
    last_price: float | None = None
    steps = 0

    try:
        while True:
            if tick_file.exists():
                mtime = tick_file.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    try:
                        tick = json.loads(tick_file.read_text(encoding="utf-8"))
                    except Exception:  # pragma: no cover - defensive
                        log.exception("invalid tick data")
                        time.sleep(0.25)
                        continue

                    ts = float(tick.get("time", time.time()))
                    price = float(
                        tick.get("bid") or tick.get("price") or tick.get("ask")
                    )
                    last_price = price
                    log.info("tick: %s", tick)

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
                        log.info("candle closed: %s", current_bar)

                        cur_eq = broker.equity()
                        if start_equity > 0 and cur_eq is not None:
                            dd = (start_equity - cur_eq) / start_equity
                            if dd >= settings.risk.max_drawdown:
                                log.error("max drawdown reached: %.2f%%", dd * 100)
                                break

                        if (
                            idx.weekday() in settings.time.blocked_weekdays
                            or idx.hour in settings.time.blocked_hours
                        ):
                            log.info("time blocked: %s", idx)
                        else:
                            sig = int(compute_signal(df, settings, "close").iloc[-1])
                            decision = agent.decide(
                                idx,
                                sig,
                                current_bar["close"],
                                settings.broker.symbol,
                                context_text,
                            )
                            log.info("decision: %s", decision)
                            if decision in ("BUY", "SELL"):
                                res = broker.market_order(
                                    decision, settings.broker.volume, price
                                )
                                log.info("order result: %s", res)

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
        log.info("KeyboardInterrupt received")
    finally:
        pos = broker.position_qty()
        if pos > 0:
            broker.market_order("SELL", pos, last_price)
        broker.close()
