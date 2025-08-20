from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Thread

import pandas as pd

from forest5.live.live_runner import run_live
from forest5.live.settings import (
    LiveSettings,
    BrokerSettings,
    DecisionSettings,
    AISettings,
    TimeSettings,
    RiskSettings,
)


def _mk_bridge(tmpdir: Path) -> Path:
    bridge = tmpdir / "forest_bridge"
    for sub in ("ticks", "state", "commands", "results"):
        (bridge / sub).mkdir(parents=True, exist_ok=True)
    return bridge


def test_soft_wait_halts_orders(tmp_path: Path, monkeypatch) -> None:
    bridge = _mk_bridge(tmp_path)
    orders: list[str] = []

    def fake_market_order(self, side, qty, price=None):
        orders.append(side)
        class R:
            id = 1
            status = "filled"
            filled_qty = qty
            avg_price = price or 1.0
        return R()

    eq_vals = [100_000.0, 100_000.0, 50_000.0, 50_000.0]

    def fake_equity(self):
        return eq_vals.pop(0)

    def fake_signal(df: pd.DataFrame, settings, price_col: str = "close") -> pd.Series:
        return pd.Series([1] * len(df), index=df.index)

    def fake_decide(self, *args, **kwargs):
        return "BUY", 1, "reason"

    monkeypatch.setattr("forest5.live.router.PaperBroker.market_order", fake_market_order)
    monkeypatch.setattr("forest5.live.router.PaperBroker.equity", fake_equity)
    monkeypatch.setattr("forest5.live.live_runner.compute_signal", fake_signal)
    monkeypatch.setattr("forest5.live.live_runner.DecisionAgent.decide", fake_decide)

    s = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=str(bridge), symbol="EURUSD", volume=1.0),
        decision=DecisionSettings(min_confluence=1),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=64, context_file=None),
        time=TimeSettings(blocked_hours=[], blocked_weekdays=[]),
        risk=RiskSettings(max_drawdown=0.1, on_drawdown={"action": "soft_wait"}),
    )

    tick_file = bridge / "ticks" / "tick.json"

    def write_tick(ts: int) -> None:
        tick_file.write_text(
            f'{{"symbol":"EURUSD","bid":1.0,"ask":1.0,"time":{ts}}}',
            encoding="utf-8",
        )
        os.utime(tick_file, None)

    write_tick(0)

    t = Thread(target=run_live, args=(s,), kwargs={"max_steps": 3, "timeout": 1})
    t.start()
    for ts in (60, 120, 180):
        time.sleep(0.5)
        write_tick(ts)
    t.join(timeout=10)
    assert not t.is_alive(), "run_live did not finish"

    assert orders == ["BUY"], orders
    assert eq_vals == [], eq_vals
