import json
import os
import sys
import types
import uuid

# mock heavy dependencies imported in package __init__
sys.modules.setdefault("numpy", types.SimpleNamespace())
sys.modules.setdefault("pandas", types.SimpleNamespace())

from forest5.live.mt4_broker import MT4Broker, OrderResult


def test_market_order(tmp_path, monkeypatch):
    bridge = tmp_path / "bridge"
    os.environ["FOREST_MT4_BRIDGE_DIR"] = str(bridge)

    broker = MT4Broker()
    broker.connect()

    fake_uuid = uuid.UUID(int=0)
    monkeypatch.setattr(uuid, "uuid4", lambda: fake_uuid)

    res_path = bridge / "results" / f"res_{fake_uuid.hex}.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(
        json.dumps({"id": 1, "status": "filled", "filled_qty": 1.0, "avg_price": 100.0}),
        encoding="utf-8",
    )

    result = broker.market_order("BUY", 1.0)

    cmd_path = bridge / "commands" / f"cmd_{fake_uuid.hex}.json"
    assert cmd_path.exists()
    assert result == OrderResult(1, "filled", 1.0, 100.0, None)


def test_state_reading(tmp_path):
    bridge = tmp_path / "bridge"
    os.environ["FOREST_MT4_BRIDGE_DIR"] = str(bridge)

    state_dir = bridge / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "position.json").write_text(json.dumps({"qty": 2.5}), encoding="utf-8")
    (state_dir / "account.json").write_text(json.dumps({"equity": 1234.5}), encoding="utf-8")

    broker = MT4Broker()
    broker.connect()

    assert broker.position_qty() == 2.5
    assert broker.equity() == 1234.5
