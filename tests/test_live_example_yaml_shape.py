from pathlib import Path
import yaml


def test_live_example_yaml_structure():
    p = Path("config/live.example.yaml")
    assert p.exists(), "config/live.example.yaml must exist"
    cfg = yaml.safe_load(p.read_text())
    for k in ["broker", "risk", "ai", "time", "decision"]:
        assert k in cfg, f"Missing '{k}' section"

    broker_type = str(cfg["broker"].get("type", "")).lower()
    assert broker_type in ("mt4", "paper", "mt5")
    if broker_type == "mt4":
        assert "bridge_dir" in cfg["broker"], "mt4 broker requires 'bridge_dir'"

    for key in ["symbol", "volume", "timeframe"]:
        assert key in cfg["broker"], f"Missing 'broker.{key}'"

    assert "enabled" in cfg["ai"], "Missing 'ai.enabled'"
    assert isinstance(cfg["ai"]["enabled"], bool), "'ai.enabled' must be boolean"

    assert "min_confluence" in cfg["decision"], "Missing 'decision.min_confluence'"
