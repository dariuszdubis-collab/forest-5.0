import json
import os
import time
import threading
from pathlib import Path

from forest5.config_live import (
    LiveSettings,
    BrokerSettings,
    DecisionSettings,
    LiveTimeSettings,
)
from forest5.config import AISettings, RiskSettings
from forest5.live.live_runner import run_live


def _mk_bridge(tmpdir: Path) -> Path:
    bridge = tmpdir
    (bridge / "ticks").mkdir()
    (bridge / "state").mkdir()
    (bridge / "commands").mkdir()
    (bridge / "results").mkdir()
    tick_path = bridge / "ticks" / "tick.json"
    tmp = tick_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"time": 1_000_000_000, "bid": 100.0}), encoding="utf-8")
    tmp.replace(tick_path)
    os.utime(tick_path, (1_000_000_000, 1_000_000_000))
    (bridge / "state" / "account.json").write_text(json.dumps({"equity": 0.0}), encoding="utf-8")
    (bridge / "state" / "position_EURUSD.json").write_text(
        json.dumps({"qty": 0.0}), encoding="utf-8"
    )
    (bridge / "commands" / "noop.json").write_text("{}", encoding="utf-8")
    (bridge / "results" / "noop.json").write_text("{}", encoding="utf-8")
    return bridge


def test_live_runner_paper_smoke(tmp_path: Path):
    bridge = _mk_bridge(tmp_path)

    settings = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=bridge, symbol="EURUSD"),
        decision=DecisionSettings(),
        ai=AISettings(),
        time=LiveTimeSettings(),
        risk=RiskSettings(),
    )

    tick_file = bridge / "ticks" / "tick.json"
    ticks = [
        {"time": 1_000_000_000, "bid": 100},
        {"time": 1_000_000_060, "bid": 101},
        {"time": 1_000_000_120, "bid": 102},
    ]

    def write_tick(tick: dict) -> None:
        tmp = tick_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(tick), encoding="utf-8")
        tmp.replace(tick_file)
        os.utime(tick_file, (tick["time"], tick["time"]))

    errors: list[Exception] = []

    def runner():
        try:
            run_live(settings, max_steps=2)
        except Exception as exc:  # pragma: no cover - fail test
            errors.append(exc)

    t = threading.Thread(target=runner)
    t.start()
    for tick in ticks[1:]:
        write_tick(tick)
        time.sleep(0.3)
    t.join(timeout=5)

    assert not t.is_alive(), "run_live did not finish"
    if errors:
        raise errors[0]


def test_live_runner_exits_on_idle_timeout(tmp_path: Path):
    bridge = _mk_bridge(tmp_path)

    settings = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=bridge, symbol="EURUSD"),
        decision=DecisionSettings(),
        ai=AISettings(),
        time=LiveTimeSettings(),
        risk=RiskSettings(),
    )

    errors: list[Exception] = []

    def runner() -> None:
        try:
            run_live(settings, timeout=1.0)
        except Exception as exc:  # pragma: no cover - fail test
            errors.append(exc)

    t = threading.Thread(target=runner)
    start = time.time()
    t.start()
    t.join(timeout=5)
    elapsed = time.time() - start

    assert not t.is_alive(), "run_live did not exit on idle timeout"
    # Give a small cushion above the requested timeout to account for scheduling
    assert elapsed < 3, f"run_live took too long: {elapsed} sec"
    if errors:
        raise errors[0]
