from pathlib import Path

def test_mql4_forestbridge_has_required_symbols():
    p = Path("mt4/ForestBridge.mq4")
    assert p.exists(), "mt4/ForestBridge.mq4 must exist in repo"
    s = p.read_text(encoding="utf-8", errors="ignore")
    required = [
        "OnInit", "OnTick", "OnTimer", "ProcessCommands",
        "OrderSend", "FileOpen", "tick.json", "commands", "results", "state"
    ]
    for r in required:
        assert r in s, f"Expected token '{r}' in ForestBridge.mq4"
