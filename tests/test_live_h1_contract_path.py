from pathlib import Path
import json
import threading
import time

import pytest

import forest5.live.live_runner as live_runner
from forest5.live.live_runner import run_live
import forest5.live.router as router
from forest5.signals.setups import TriggeredSignal
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
    (bridge / "ticks" / "tick.json").write_text(
        '{"symbol":"EURUSD","bid":1.0,"ask":1.0,"time":0}',
        encoding="utf-8",
    )
    (bridge / "state" / "account.json").write_text('{"equity":10000}', encoding="utf-8")
    (bridge / "state" / "position_EURUSD.json").write_text('{"qty":0}', encoding="utf-8")
    return bridge


def _update_ticks(path: Path, ticks: list[dict], delay: float = 0.1) -> None:
    for tick in ticks:
        time.sleep(delay)
        path.write_text(json.dumps(tick), encoding="utf-8")


def test_h1_contract_arm_and_trigger(tmp_path: Path, monkeypatch):
    bridge = _mk_bridge(tmp_path)
    tick_path = bridge / "ticks" / "tick.json"

    orig_append = live_runner.append_bar_and_signal

    calls = {"n": 0}

    def fake_append(df, bar, settings, **kwargs):
        orig_append(df, bar, settings)
        calls["n"] += 1
        return 1 if calls["n"] == 1 else 0

    monkeypatch.setattr(live_runner, "append_bar_and_signal", fake_append)

    class TriggerRegistry:
        last = None

        def __init__(self, *args, **kwargs):
            self.armed = False
            self.arm_index: int | None = None
            self.trigger_index: int | None = None
            self.check_indices: list[int] = []
            TriggerRegistry.last = self

        def arm(self, key, index, signal, *, ctx=None):
            self.armed = True
            self.arm_index = index

        def check(self, *, index, price, ctx=None):
            self.check_indices.append(index)
            if self.armed and index >= (self.arm_index or 0):
                self.armed = False
                self.trigger_index = index
                return TriggeredSignal(
                    setup_id="s1",
                    action="BUY",
                    entry=1.0,
                    sl=0.0,
                    tp=0.0,
                )
            return None

    monkeypatch.setattr(live_runner, "SetupRegistry", TriggerRegistry)

    created: dict[str, object] = {}

    class FakeBroker:
        def __init__(self):
            created["inst"] = self
            self.ticks_dir = None
            self.connected = False

        def connect(self):
            self.connected = True

        def close(self):
            pass

        def set_price(self, price: float) -> None:  # pragma: no cover - interface
            pass

        def market_order(self, side, qty, price, *, entry=None, sl=None, tp=None):
            created.setdefault("orders", []).append((side, qty, price))
            return router.OrderResult(1, "filled", qty, price)

        def position_qty(self):
            return 0.0

        def equity(self):
            return 10000.0

    monkeypatch.setattr(router, "PaperBroker", FakeBroker)

    s = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=str(bridge), symbol="EURUSD", volume=0.01),
        decision=DecisionSettings(min_confluence=0.5),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=64, context_file=None),
        time=TimeSettings(),
        risk=RiskSettings(max_drawdown=0.5),
    )

    t = threading.Thread(
        target=_update_ticks,
        args=(
            tick_path,
            [
                {"symbol": "EURUSD", "bid": 1.0, "ask": 1.0, "time": 61},
                {"symbol": "EURUSD", "bid": 1.0, "ask": 1.0, "time": 121},
            ],
            0.5,
        ),
    )
    t.start()
    run_live(s, max_steps=2, timeout=2.0)
    t.join()

    broker = created.get("inst")
    assert isinstance(broker, FakeBroker)
    assert broker.ticks_dir == bridge / "ticks"
    # Bar N should arm and bar N+1 should trigger and route to the broker
    registry = TriggerRegistry.last
    assert registry is not None
    assert registry.arm_index == 1
    assert registry.check_indices == [0, 1]
    assert registry.trigger_index == 1
    assert created.get("orders") == [("BUY", pytest.approx(0.009), 1.0)]
