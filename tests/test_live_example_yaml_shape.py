from pathlib import Path
import yaml


def test_live_example_yaml_structure():
    p = Path("config/live.example.yaml")
    assert p.exists(), "config/live.example.yaml must exist"
    cfg = yaml.safe_load(p.read_text())
    for k in ["broker", "strategy", "risk", "ai", "time"]:
        assert k in cfg, f"Missing '{k}' section"
    assert cfg["broker"]["type"] in ("MT4", "Paper", "MT5")
    if cfg["broker"]["type"] == "MT4":
        assert "bridge_dir" in cfg["broker"], "MT4 broker requires 'bridge_dir'"
    assert "symbol" in cfg["broker"]
    assert isinstance(cfg["ai"].get("enabled", False), bool)
