from pathlib import Path
from forest5.config_live import LiveSettings


def test_live_settings_from_yaml(tmp_path: Path):
    p = tmp_path / "cfg_live.yaml"
    p.write_text(
        "broker:\n"
        "  type: mt4\n"
        "  bridge_dir: /tmp/bridge\n"
        "  symbol: EURUSD\n"
        "  volume: 1.5\n"
        "strategy:\n"
        "  timeframe: 1m\n"
        "  fast: 10\n"
        "  slow: 30\n"
        "ai:\n"
        "  enabled: true\n"
        "time:\n"
        "  blocked_weekdays: [5, 6]\n"
        "  blocked_hours: [0]\n",
        encoding="utf-8",
    )
    s = LiveSettings.from_file(p)
    assert s.broker.symbol == "EURUSD"
    assert s.broker.volume == 1.5
    assert s.strategy.fast == 10
    assert s.strategy.timeframe == "1m"
    assert s.time.blocked_weekdays == [5, 6]
