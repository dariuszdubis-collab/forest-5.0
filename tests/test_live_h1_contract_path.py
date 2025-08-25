import json
import threading
import time
from types import SimpleNamespace

import pandas as pd

from forest5.live.live_runner import run_live, append_bar_and_signal
from forest5.live.settings import (
    LiveSettings,
    BrokerSettings,
    DecisionSettings,
    AISettings,
    TimeSettings,
    RiskSettings,
)
from forest5.config_live import StrategySettings
from forest5.signals.contract import TechnicalSignal


def test_h1_contract_arm_and_trigger(monkeypatch, tmp_path, capsys):
    calls = {"count": 0}
    events: list[str] = []

    def fake_compute_signal(df, strategy, ctx=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return TechnicalSignal(action="BUY", entry=1.1, sl=0.9, tp=1.3)
        return TechnicalSignal()

    monkeypatch.setattr("forest5.live.live_runner.compute_signal", fake_compute_signal)
    monkeypatch.setattr(
        "forest5.live.live_runner.log_event", lambda e, ctx=None, **f: events.append(e)
    )

    class FakeSetupRegistry:
        def __init__(self):
            self.sig: TechnicalSignal | None = None

        def arm(self, key, bar_index, contract, ctx=None):
            self.sig = contract

        def check(self, key, bar_index, price, ctx=None):
            if self.sig and price >= self.sig.entry:
                sig = self.sig
                self.sig = None
                return sig
            return None

    def fake_setup_registry():
        reg = FakeSetupRegistry()
        return reg

    monkeypatch.setattr("forest5.live.live_runner.SetupRegistry", fake_setup_registry)

    class FakeAgent:
        def __init__(self, *a, **k):
            pass

        def decide(self, idx, sig, price, symbol, context):
            action = sig.action if isinstance(sig, TechnicalSignal) else "WAIT"
            return SimpleNamespace(decision=action, weight_sum=1.0, votes={}, reason="")

    monkeypatch.setattr("forest5.live.live_runner.DecisionAgent", FakeAgent)

    broker_holder: dict[str, any] = {}

    class FakeBroker:
        def __init__(self):
            self.ticks_dir = tmp_path / "ticks"
            self.orders: list[tuple] = []
            broker_holder["instance"] = self

        def connect(self):
            pass

        def equity(self):
            return 1000.0

        def market_order(self, side, qty, price, sl=None, tp=None):
            self.orders.append((side, qty, price, sl, tp))
            return SimpleNamespace(
                filled_qty=qty,
                avg_price=price,
                status="FILLED",
                error=None,
                id="1",
            )

        def position_qty(self):
            return 0

        def close(self):
            pass

    monkeypatch.setattr("forest5.live.router.PaperBroker", FakeBroker)

    monkeypatch.setattr("time.sleep", lambda _: None)

    settings = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=tmp_path, symbol="EURUSD", volume=1.0),
        strategy=StrategySettings(name="h1_ema_rsi_atr", timeframe="1m"),
        risk=RiskSettings(max_drawdown=1.0),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=0, context_file=None),
        decision=DecisionSettings(min_confluence=0),
        time=TimeSettings(),
    )

    tick_dir = tmp_path / "ticks"
    tick_dir.mkdir()
    tick_file = tick_dir / "tick.json"

    def write_tick(t):
        tick_file.write_text(json.dumps(t))

    def runner():
        run_live(settings, max_steps=2, timeout=0.5)

    t = threading.Thread(target=runner)
    t.start()
    write_tick({"time": 0, "bid": 1.0})
    time.sleep(0.01)
    write_tick({"time": 60, "bid": 1.0})
    time.sleep(0.01)
    write_tick({"time": 61, "bid": 1.2})
    time.sleep(0.01)
    write_tick({"time": 120, "bid": 1.2})
    t.join()

    _ = capsys.readouterr().out
    assert "setup_arm" in events
    assert "setup_trigger" in events
    broker = broker_holder["instance"]
    assert broker.orders[0][2] == 1.1
    assert broker.orders[0][3] == 0.9
    assert broker.orders[0][4] == 1.3


def test_append_bar_uses_compute_signal_compat_for_ema_cross(monkeypatch):
    calls = {"compat": 0}

    def fake_compat(df, settings, price_col="close"):
        calls["compat"] += 1
        return pd.Series([0], index=df.index)

    monkeypatch.setattr("forest5.live.live_runner.compute_signal_compat", fake_compat)

    def fake_compute(*args, **kwargs):
        raise AssertionError("compute_signal called")

    monkeypatch.setattr("forest5.live.live_runner.compute_signal", fake_compute)

    settings = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=".", symbol="EURUSD", volume=1.0),
        strategy=StrategySettings(name="ema_cross", timeframe="1m"),
        risk=RiskSettings(max_drawdown=1.0),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=0, context_file=None),
        decision=DecisionSettings(min_confluence=0),
        time=TimeSettings(),
    )

    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    bar = {"start": 0, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}
    append_bar_and_signal(df, bar, settings)
    assert calls["compat"] == 1
