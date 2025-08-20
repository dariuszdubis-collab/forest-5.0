from pathlib import Path

import timeit

import pandas as pd

from forest5.live.live_runner import run_live, _append_bar_and_signal
from forest5.signals.factory import compute_signal
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
    (bridge / "state" / "account.json").write_text(
        '{"equity":10000}', encoding="utf-8"
    )
    (bridge / "state" / "position_EURUSD.json").write_text(
        '{"qty":0}', encoding="utf-8"
    )
    return bridge


def test_run_live_paper(tmp_path: Path, capfd):
    bridge = _mk_bridge(tmp_path)
    s = LiveSettings(
        broker=BrokerSettings(
            type="paper", bridge_dir=str(bridge), symbol="EURUSD", volume=0.01
        ),
        decision=DecisionSettings(min_confluence=1),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=64, context_file=None),
        time=TimeSettings(blocked_hours=[], blocked_weekdays=[]),
        risk=RiskSettings(max_drawdown=0.5),
    )
    run_live(s, max_steps=2)
    out = capfd.readouterr().out
    assert "idle_timeout_reached" in out


def test_incremental_signal_perf():
    settings = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=".", symbol="EURUSD", volume=0.01),
        decision=DecisionSettings(min_confluence=1),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=64, context_file=None),
        time=TimeSettings(blocked_hours=[], blocked_weekdays=[]),
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
            compute_signal(df, settings, "close").iloc[-1]

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
