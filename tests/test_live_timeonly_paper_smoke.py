from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from forest5.decision import DecisionAgent, DecisionResult
from forest5.live.live_runner import run_live
from forest5.live.settings import LiveSettings, BrokerSettings, DecisionSettings, TimeSettings


def _write_min_bridge(tmp_path: Path) -> Path:
    bridge = tmp_path / "bridge"
    for sub in ("ticks", "state", "commands", "results"):
        (bridge / sub).mkdir(parents=True, exist_ok=True)
    (bridge / "ticks" / "tick.json").write_text(
        json.dumps({"symbol": "EURUSD", "bid": 1.0, "ask": 1.0, "time": 0}),
        encoding="utf-8",
    )
    (bridge / "state" / "account.json").write_text('{"equity":10000}', encoding="utf-8")
    (bridge / "state" / "position_EURUSD.json").write_text('{"qty":0}', encoding="utf-8")
    return bridge


@pytest.mark.timeonly
@pytest.mark.e2e
def test_live_timeonly_paper_smoke(tmp_path: Path, monkeypatch, capfd) -> None:
    bridge = _write_min_bridge(tmp_path)

    def fake_decide(self, *args, **kwargs):  # pragma: no cover - simple stub
        return DecisionResult("BUY", 0.5, {"tech": 1}, "")

    monkeypatch.setattr(DecisionAgent, "decide", fake_decide)

    settings = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=str(bridge), symbol="EURUSD", volume=0.01),
        decision=DecisionSettings(min_confluence=1.0),
        time=TimeSettings(),
    )

    tick_file = bridge / "ticks" / "tick.json"

    def runner() -> None:
        run_live(settings, max_steps=1)

    def feeder() -> None:
        time.sleep(1.1)
        tick_file.write_text(
            json.dumps({"symbol": "EURUSD", "bid": 1.0, "ask": 1.0, "time": 61}),
            encoding="utf-8",
        )

    feeder_thread = threading.Thread(target=feeder)
    feeder_thread.start()
    run_live(settings, max_steps=1)
    feeder_thread.join()

    out = capfd.readouterr().out
    assert '"order_result"' in out
    assert '"qty": 0.005' in out
