from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from forest5.live.live_runner import run_live
from forest5.live.settings import LiveSettings, BrokerSettings, DecisionSettings, TimeSettings
from forest5.config_live import LiveTimeModelSettings


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
def test_live_timeonly_paper_smoke(tmp_path: Path, caplog) -> None:
    bridge = _write_min_bridge(tmp_path)

    model = tmp_path / "time_model.json"
    model.write_text(
        json.dumps({"quantile_gates": {"0": [0.5, 1.5]}, "q_low": 0.25, "q_high": 0.75}),
        encoding="utf-8",
    )

    settings = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=str(bridge), symbol="EURUSD", volume=0.01),
        decision=DecisionSettings(min_confluence=2),
        time=TimeSettings(model=LiveTimeModelSettings(enabled=True, path=str(model))),
    )

    tick_file = bridge / "ticks" / "tick.json"

    def runner() -> None:
        run_live(settings, max_steps=1)

    caplog.set_level("INFO")
    t = threading.Thread(target=runner)
    t.start()
    time.sleep(0.5)
    tick_file.write_text(
        json.dumps({"symbol": "EURUSD", "bid": 1.0, "ask": 1.0, "time": 61}),
        encoding="utf-8",
    )
    t.join(timeout=5)
    assert not t.is_alive()

    assert all("order result" not in rec.message for rec in caplog.records)
