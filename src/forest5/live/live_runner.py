from __future__ import annotations

import json
import logging
import time

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from ..config import RiskSettings, StrategySettings
from ..decision import DecisionAgent
from ..signals.factory import compute_signal
from ..utils.timeframes import _TF_MINUTES, normalize_timeframe
from .mt4_broker import MT4Broker

log = logging.getLogger(__name__)


class LiveStrategySettings(StrategySettings):
    timeframe: str = "1m"

    @field_validator("timeframe")
    @classmethod
    def _norm_tf(cls, v: str) -> str:
        return normalize_timeframe(v)


class LiveSettings(BaseModel):
    symbol: str = "SYMBOL"
    strategy: LiveStrategySettings = Field(default_factory=LiveStrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)


def run_live(settings: LiveSettings) -> None:
    broker = MT4Broker()
    broker.connect()

    agent = DecisionAgent(router=broker)

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
                        tick.get("bid")
                        or tick.get("price")
                        or tick.get("ask")
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
                        sig = int(compute_signal(df, settings, "close").iloc[-1])
                        decision = agent.decide(idx, sig, current_bar["close"], settings.symbol)
                        log.info("decision: %s", decision)
                        if decision in ("BUY", "SELL"):
                            res = broker.market_order(decision, 1.0, price)
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
