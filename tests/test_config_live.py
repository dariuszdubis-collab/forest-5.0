from pathlib import Path

import pytest

from forest5.config_live import LiveSettings, LiveTimeModelSettings, LiveTimeSettings


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
    assert s.broker.symbol == "EURUSD"  # nosec B101
    assert s.broker.volume == 1.5  # nosec B101
    assert s.strategy.fast == 10  # nosec B101
    assert s.strategy.timeframe == "1m"  # nosec B101
    assert s.time.blocked_weekdays == [5, 6]  # nosec B101


def test_live_settings_ai_context_file_resolved(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    ctx = cfg_dir / "ctx.txt"
    ctx.write_text("hi", encoding="utf-8")
    cfg = cfg_dir / "cfg_live.yaml"
    cfg.write_text(
        "broker:\n  type: mt4\n" "ai:\n  context_file: ctx.txt\n",
        encoding="utf-8",
    )

    settings = LiveSettings.from_file(cfg)

    assert settings.ai.context_file == str(ctx)  # nosec B101


def test_live_time_model_quantile_valid():
    m = LiveTimeModelSettings(q_low=0.0, q_high=1.0)
    assert m.q_low == 0.0 and m.q_high == 1.0  # nosec B101


@pytest.mark.parametrize(
    "q_low,q_high",
    [
        (-0.1, 0.9),
        (0.2, 1.1),
        (0.5, 0.4),
        (0.5, 0.5),
    ],
)
def test_live_time_model_quantile_invalid(q_low: float, q_high: float):
    with pytest.raises(ValueError, match="0.0 <= q_low < q_high <= 1.0"):
        LiveTimeModelSettings(q_low=q_low, q_high=q_high)


def test_live_time_settings_invalid_weekday():
    with pytest.raises(ValueError, match="blocked_weekdays"):
        LiveTimeSettings(blocked_weekdays=[-1, 7])


def test_live_time_settings_invalid_hour():
    with pytest.raises(ValueError, match="blocked_hours"):
        LiveTimeSettings(blocked_hours=[-1, 24])
