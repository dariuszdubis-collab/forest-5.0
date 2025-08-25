from pathlib import Path

import json
import threading
import time
import timeit

import pandas as pd

import forest5.live.live_runner as live_runner
from forest5.live.live_runner import run_live, _append_bar_and_signal
from forest5.signals.compat import compute_signal_compat
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


def _update_tick(path: Path, tick: dict, delay: float = 0.1) -> None:
    time.sleep(delay)
    path.write_text(json.dumps(tick), encoding="utf-8")


def test_run_live_paper(tmp_path: Path, capfd):
    bridge = _mk_bridge(tmp_path)
    s = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=str(bridge), symbol="EURUSD", volume=0.01),
        decision=DecisionSettings(min_confluence=1),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=64, context_file=None),
        time=TimeSettings(),
        risk=RiskSettings(max_drawdown=0.5),
    )
    run_live(s, max_steps=2)
    out = capfd.readouterr().out
    assert "idle_timeout_reached" in out


def test_triggered_setup_executes(tmp_path: Path, capfd, monkeypatch):
    bridge = _mk_bridge(tmp_path)
    tick_path = bridge / "ticks" / "tick.json"

    orig_append = live_runner.append_bar_and_signal

    def fake_append(df, bar, settings):
        orig_append(df, bar, settings)
        return 1

    monkeypatch.setattr(live_runner, "append_bar_and_signal", fake_append)

    class TriggerRegistry:
        def __init__(self, *args, **kwargs):
            self.armed = False

        def arm(self, signal, *, expiry=None, ctx=None):
            self.armed = True

        def check(self, key, index, high, low):
            if self.armed:
                self.armed = False
                return TechnicalSignal(
                    timeframe="1m",
                    action="BUY",
                    entry=1.0,
                    horizon_minutes=1,
                    technical_score=1.0,
                    confidence_tech=1.0,
                )
            return None

    monkeypatch.setattr(live_runner, "SetupRegistry", TriggerRegistry)

    s = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=str(bridge), symbol="EURUSD", volume=0.01),
        decision=DecisionSettings(min_confluence=1),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=64, context_file=None),
        time=TimeSettings(),
        risk=RiskSettings(max_drawdown=0.5),
    )

    t = threading.Thread(
        target=_update_tick,
        args=(
            tick_path,
            {"symbol": "EURUSD", "bid": 1.0, "ask": 1.0, "time": 61},
            0.1,
        ),
    )
    t.start()
    run_live(s, max_steps=2, timeout=0.5)
    t.join()
    out = capfd.readouterr().out
    assert "order_result" in out


def test_setup_expires_without_trigger(tmp_path: Path, capfd, monkeypatch):
    bridge = _mk_bridge(tmp_path)
    tick_path = bridge / "ticks" / "tick.json"

    orig_append = live_runner.append_bar_and_signal

    def fake_append(df, bar, settings):
        orig_append(df, bar, settings)
        return 1

    monkeypatch.setattr(live_runner, "append_bar_and_signal", fake_append)

    class NoTriggerRegistry:
        def __init__(self, *args, **kwargs):
            pass

        def arm(self, signal, *, expiry=None, ctx=None):
            pass

        def check(self, key, index, high, low):
            return None

    monkeypatch.setattr(live_runner, "SetupRegistry", NoTriggerRegistry)

    s = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=str(bridge), symbol="EURUSD", volume=0.01),
        decision=DecisionSettings(min_confluence=1),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=64, context_file=None),
        time=TimeSettings(),
        risk=RiskSettings(max_drawdown=0.5),
    )

    t = threading.Thread(
        target=_update_tick,
        args=(
            tick_path,
            {"symbol": "EURUSD", "bid": 1.0, "ask": 1.0, "time": 61},
            0.1,
        ),
    )
    t.start()
    run_live(s, max_steps=2, timeout=0.5)
    t.join()
    out = capfd.readouterr().out
    assert "order_result" not in out


def test_incremental_signal_perf():
    settings = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=".", symbol="EURUSD", volume=0.01),
        decision=DecisionSettings(min_confluence=1),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=64, context_file=None),
        time=TimeSettings(),
        risk=RiskSettings(max_drawdown=0.5),
    )
    N = 200

    def naive() -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close"])
        bar = {"start": 0, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}
        for i in range(N):
            bar["start"] = i
            bar["open"] = bar["high"] = bar["low"] = bar["close"] = 1 + 0.0001 * i
            idx = pd.to_datetime(bar["start"], unit="s")
            df.loc[idx] = [bar["open"], bar["high"], bar["low"], bar["close"]]
            compute_signal_compat(df, settings, "close").iloc[-1]

    def incremental() -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close"])
        bar = {"start": 0, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}
        for i in range(N):
            bar["start"] = i
            bar["open"] = bar["high"] = bar["low"] = bar["close"] = 1 + 0.0001 * i
            _append_bar_and_signal(df, bar, settings)

    t_naive = timeit.timeit(naive, number=3)
    t_inc = timeit.timeit(incremental, number=3)
    assert t_inc < t_naive
