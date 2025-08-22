import json
import threading
from pathlib import Path

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
from forest5.config import OnDrawdownSettings


def _mk_bridge(tmpdir: Path) -> Path:
    bridge = tmpdir / "forest_bridge"
    for sub in ("ticks", "state", "commands", "results"):
        (bridge / sub).mkdir(parents=True, exist_ok=True)
    (bridge / "state" / "account.json").write_text('{"equity":10000}', encoding="utf-8")
    (bridge / "state" / "position_EURUSD.json").write_text('{"qty":0}', encoding="utf-8")
    return bridge


def test_run_live_soft_wait(tmp_path: Path, monkeypatch) -> None:
    bridge = _mk_bridge(tmp_path)
    ready = threading.Event()
    tick_event = threading.Event()
    tick_file = bridge / "ticks" / "tick.json"

    def fake_append_bar_and_signal(df, bar, settings):
        idx = pd.to_datetime(bar["start"], unit="s")
        df.loc[idx, ["open", "high", "low", "close"]] = [
            bar["open"],
            bar["high"],
            bar["low"],
            bar["close"],
        ]
        return pd.Series([1], index=df.index)

    monkeypatch.setattr(
        "forest5.live.live_runner._append_bar_and_signal", fake_append_bar_and_signal
    )

    orig_log = run_live.__globals__["log"].info

    def log_info(msg: str, **kwargs) -> None:
        if msg == "tick":
            tick_event.set()
        orig_log(msg, **kwargs)

    monkeypatch.setattr("forest5.live.live_runner.log.info", log_info)

    orig_exists = Path.exists

    def exists(self: Path) -> bool:
        if self == tick_file and not ready.is_set():
            ready.set()
        return orig_exists(self)

    monkeypatch.setattr(Path, "exists", exists)

    settings = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=str(bridge), symbol="EURUSD", volume=1),
        decision=DecisionSettings(min_confluence=1),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=64, context_file=None),
        time=TimeSettings(blocked_hours=[], blocked_weekdays=[]),
        risk=RiskSettings(
            max_drawdown=0.01,
            on_drawdown=OnDrawdownSettings(action="soft_wait"),
        ),
    )

    t = threading.Thread(target=lambda: run_live(settings, max_steps=1, timeout=2.0))
    t.start()
    assert ready.wait(timeout=5)

    tick_file.write_text(
        json.dumps({"symbol": "EURUSD", "bid": 1.0, "ask": 1.0, "time": 0}),
        encoding="utf-8",
    )
    assert tick_event.wait(timeout=5)

    t.join(timeout=5)
    assert not t.is_alive()
