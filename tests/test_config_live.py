from pathlib import Path

import pytest

from pydantic import ValidationError

from forest5.config.loader import load_live_settings
from forest5.config_live import LiveTimeModelSettings, validate_live_config
from forest5.config.strategy import BaseStrategySettings, PatternSettings


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
        "  enabled: false\n"
        "time:\n"
        "  model:\n"
        "    enabled: true\n"
        "    path: model.onnx\n",
        encoding="utf-8",
    )
    s = load_live_settings(p)
    assert s.broker.symbol == "EURUSD"  # nosec B101
    assert s.broker.volume == 1.5  # nosec B101
    assert s.strategy.fast == 10  # nosec B101
    assert s.strategy.timeframe == "1m"  # nosec B101
    assert isinstance(s.strategy, BaseStrategySettings)
    assert s.time.model.enabled is True  # nosec B101
    assert s.time.model.path == p.parent / "model.onnx"  # nosec B101
    assert s.risk.on_drawdown.action == "halt"  # nosec B101
    assert s.decision.tie_epsilon == 0.05  # nosec B101
    assert s.decision.weights.tech == 1.0  # nosec B101
    assert s.decision.weights.ai == 0.5  # nosec B101
    assert s.decision.tech.default_conf_int == 0.50  # nosec B101
    assert s.decision.tech.conf_floor == 0.20  # nosec B101
    assert s.decision.tech.conf_cap == 0.90  # nosec B101
    default_patterns = PatternSettings().model_dump()
    assert s.strategy.patterns.model_dump() == default_patterns  # nosec B101
    assert s.time.primary_signal.patterns.model_dump() == default_patterns  # nosec B101


def test_live_settings_ai_context_file_resolved(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    ctx = cfg_dir / "ctx.txt"
    ctx.write_text("hi", encoding="utf-8")
    cfg = cfg_dir / "cfg_live.yaml"
    cfg.write_text(
        "broker:\n  type: mt4\n  symbol: EURUSD\n  volume: 1.0\n" "ai:\n  context_file: ctx.txt\n",
        encoding="utf-8",
    )

    settings = load_live_settings(cfg)

    assert settings.ai.context_file == str(ctx)  # nosec B101
    assert settings.risk.on_drawdown.action == "halt"  # nosec B101


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
    with pytest.raises(ValidationError, match="0.0 <= q_low < q_high <= 1.0"):
        LiveTimeModelSettings(q_low=q_low, q_high=q_high)


def test_live_settings_h1_ema_rsi_atr(tmp_path: Path):
    p = tmp_path / "cfg_live.yaml"
    p.write_text(
        "broker:\n  type: mt4\n  symbol: EURUSD\n  volume: 1.0\n"
        "strategy:\n  name: h1_ema_rsi_atr\n  compat_int: 42\n  params:\n    ema_fast: 21\n    ema_slow: 55\n",
        encoding="utf-8",
    )

    s = load_live_settings(p)

    assert s.strategy.name == "h1_ema_rsi_atr"  # nosec B101
    assert s.strategy.compat_int == 42  # nosec B101
    assert s.strategy.params["ema_fast"] == 21  # nosec B101


def test_validate_live_config_rejects_bad_enum(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "broker:\n  type: nope\n  symbol: EURUSD\n  bridge_dir: .\n"
        "risk:\n  risk_per_trade: 0.1\n  max_drawdown: 0.2\n",
        encoding="utf-8",
    )
    ok, info = validate_live_config(p)
    assert not ok
    assert "broker.type" in info["error"] or "Unsupported" in info["error"]


def test_validate_live_config_requires_ai_context(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "broker:\n  type: mt4\n  symbol: EURUSD\n  bridge_dir: .\n"
        "risk:\n  risk_per_trade: 0.1\n  max_drawdown: 0.2\n"
        "ai:\n  enabled: true\n",
        encoding="utf-8",
    )
    ok, info = validate_live_config(p)
    assert not ok
    assert "ai.context_file" in info["error"]
