import json
import time
import threading

from forest5.config_live import LiveSettings, BrokerSettings
from forest5.live.live_runner import run_live



def test_live_runner_paper_smoke(tmp_path):
    bridge_dir = tmp_path
    ticks_dir = bridge_dir / "ticks"
    ticks_dir.mkdir()

    settings = LiveSettings(
        broker=BrokerSettings(
            type="paper",
            bridge_dir=bridge_dir,
            symbol="EURUSD",
        )
    )

    tick_file = ticks_dir / "tick.json"

    def writer():
        start = 1_000_000_000
        time.sleep(0.5)
        for i in range(4):
            tick = {"time": start + i * 60, "bid": 100 + i}
            tick_file.write_text(json.dumps(tick))
            time.sleep(0.2)

    errors: list[Exception] = []

    def runner():
        try:
            run_live(settings, max_steps=2)
        except Exception as exc:  # pragma: no cover - fail test
            errors.append(exc)

    t = threading.Thread(target=runner)
    t.start()
    writer()
    t.join(timeout=5)
    assert not t.is_alive(), "run_live did not finish"
    if errors:
        raise errors[0]
