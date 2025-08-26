import json
import threading
from pathlib import Path

import pandas as pd
import forest5.live.live_runner as live_runner
from forest5.live.live_runner import run_live
from forest5.signals.contract import TechnicalSignal
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


def test_live_ttl_minutes_expire(tmp_path, monkeypatch):
    bridge = _mk_bridge(tmp_path)
    tick_path = bridge / "ticks" / "tick.json"

    class StubRegistry:
        def __init__(self):
            self.expired = False
            self.arm_time = None
            self.ttl = None

        def arm(self, key, index, signal, *, bar_time, ttl_minutes=None, ctx=None):
            self.arm_time = pd.to_datetime(bar_time)
            self.ttl = ttl_minutes

        def check(self, *, index, price, now, ctx=None):
            if self.ttl is not None and (now - self.arm_time).total_seconds() / 60 >= self.ttl:
                self.expired = True
            return None

    registry = StubRegistry()
    monkeypatch.setattr(live_runner, "SetupRegistry", lambda *a, **k: registry)

    def fake_append(df, bar, settings, *, setup_registry=None, ctx=None):
        idx = pd.to_datetime(bar["start"], unit="s")
        df.loc[idx, ["open", "high", "low", "close"]] = [
            bar["open"],
            bar["high"],
            bar["low"],
            bar["close"],
        ]
        sig = TechnicalSignal(action="BUY", entry=bar["close"], sl=0.0, tp=0.0)
        setup_registry.arm("s1", len(df) - 1, sig, bar_time=idx, ttl_minutes=1, ctx=ctx)
        return 0

    monkeypatch.setattr(live_runner, "append_bar_and_signal", fake_append)

    s = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=str(bridge), symbol="EURUSD", volume=0.01),
        decision=DecisionSettings(min_confluence=0.0),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=64, context_file=None),
        time=TimeSettings(),
        risk=RiskSettings(max_drawdown=0.5),
    )

    def update_tick():
        import time as _t

        _t.sleep(0.1)
        tick_path.write_text(
            json.dumps({"symbol": "EURUSD", "bid": 1.0, "ask": 1.0, "time": 120}),
            encoding="utf-8",
        )

    t = threading.Thread(target=update_tick)
    t.start()
    run_live(s, max_steps=1, timeout=1)
    t.join()

    assert registry.expired
