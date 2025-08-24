from pathlib import Path
import json

from forest5.live.mt4_broker import MT4Broker

def test_command_contains_sl_tp(tmp_path: Path):
    bridge = tmp_path / "bridge"
    br = MT4Broker(bridge_dir=bridge, symbol="EURUSD", timeout_sec=0.2)
    br.connect()
    br.market_order("BUY", 0.1, price=1.2345, sl=1.2300, tp=1.2400)
    cmds = list((bridge / "commands").glob("cmd_*.json"))
    assert cmds, "no command file written"
    data = json.loads(cmds[0].read_text(encoding="utf-8"))
    assert data["sl"] == 1.2300
    assert data["tp"] == 1.2400

