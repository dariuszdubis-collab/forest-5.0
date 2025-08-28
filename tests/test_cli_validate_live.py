import json
from pathlib import Path

from forest5.cli import main


def _write_basic_config(tmp_path: Path, bridge: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
broker:
  type: "paper"
  bridge_dir: "{bridge}"
  symbol: "EURUSD"
risk:
  max_drawdown: 0.20
""",
        encoding="utf-8",
    )
    return cfg


def test_cli_validate_live_ok(tmp_path):
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "symbol_specs.json").write_text(
        json.dumps({"digits": 5, "point": 0.00001, "stop_level": 10}),
        encoding="utf-8",
    )
    cfg = _write_basic_config(tmp_path, bridge)
    rc = main(["validate", "live-config", "--yaml", str(cfg)])
    assert rc == 0


def test_cli_validate_live_missing_fields(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("broker:\n  symbol: 'EURUSD'\n", encoding="utf-8")
    rc = main(["validate", "live-config", "--yaml", str(cfg)])
    assert rc == 2


def test_live_config_bad_missing_broker_type_fails(tmp_path):
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    cfg = tmp_path / "bad_type.yaml"
    cfg.write_text(
        f"""
broker:
  bridge_dir: "{bridge}"
  symbol: "EURUSD"
risk:
  risk_per_trade: 0.01
  max_drawdown: 0.2
""",
        encoding="utf-8",
    )
    rc = main(["validate", "live-config", "--yaml", str(cfg)])
    assert rc == 2


def test_live_config_ai_enabled_missing_context_fails(tmp_path):
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    cfg = tmp_path / "ai.yaml"
    cfg.write_text(
        f"""
broker:
  type: "paper"
  bridge_dir: "{bridge}"
  symbol: "EURUSD"
risk:
  risk_per_trade: 0.01
  max_drawdown: 0.2
ai:
  enabled: true
""",
        encoding="utf-8",
    )
    rc = main(["validate", "live-config", "--yaml", str(cfg)])
    assert rc == 2


def test_live_config_bad_quantiles_fails(tmp_path):
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    cfg = tmp_path / "quant.yaml"
    cfg.write_text(
        f"""
broker:
  type: "paper"
  bridge_dir: "{bridge}"
  symbol: "EURUSD"
risk:
  risk_per_trade: 0.01
  max_drawdown: 0.2
time:
  model:
    enabled: true
    q_low: 0.6
    q_high: 0.4
""",
        encoding="utf-8",
    )
    rc = main(["validate", "live-config", "--yaml", str(cfg)])
    assert rc == 2
