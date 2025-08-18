from __future__ import annotations

import json
import logging
import time

import pandas as pd

from ..config_live import LiveSettings
from ..decision import DecisionAgent, DecisionConfig
from ..signals.factory import compute_signal
from ..utils.timeframes import _TF_MINUTES
from .mt4_broker import MT4Broker

log = logging.getLogger(__name__)


def run_live(settings: LiveSettings) -> None:
    if settings.broker.type.lower() != "mt4":
        raise ValueError(f"unsupported broker type: {settings.broker.type}")
    broker = MT4Broker(settings.broker.bridge_dir, symbol=settings.broker.symbol)
    broker.connect()

    agent = DecisionAgent(
        router=broker, config=DecisionConfig(use_ai=settings.ai.enabled)
    )

    tf = settings.strategy.timeframe
    bar_sec = _TF_MINUTES[tf] * 60

    tick_file = broker.ticks_dir / "tick.json"
    last_mtime = 0.0

    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    current_bar: dict | None = None
    last_price: float | None = None

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
            time.sleep(0.25)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received")
        pos = broker.position_qty()
        if pos > 0:
            broker.market_order("SELL", pos, last_price)
        broker.close()
