import json
from pathlib import Path

import pytest

from forest5.live.mt4_broker import MT4Broker
from forest5.utils.log import E_BROKER_ADJUSTED_STOPS


def test_stop_level_clamps_sl_tp(tmp_path: Path, capfd) -> None:
    bridge = tmp_path / "bridge"
    state = bridge / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "stop_level_EURUSD.json").write_text(
        json.dumps({"stop_level": 0.0005}), encoding="utf-8"
    )

    br = MT4Broker(bridge_dir=bridge, symbol="EURUSD", timeout_sec=0.2)
    br.connect()
    br.market_order("BUY", 0.1, price=1.2345, sl=1.2343, tp=1.2347)

    cmd = next((bridge / "commands").glob("cmd_*.json"))
    data = json.loads(cmd.read_text(encoding="utf-8"))
    assert data["sl"] == pytest.approx(1.2340)
    assert data["tp"] == pytest.approx(1.2350)
    captured = capfd.readouterr()
    assert E_BROKER_ADJUSTED_STOPS in captured.out


def test_wait_for_result_returns_adjusted_stops(tmp_path: Path) -> None:
    bridge = tmp_path / "bridge"
    br = MT4Broker(bridge_dir=bridge, symbol="EURUSD", timeout_sec=0.2)
    br.connect()

    uid = "abc123"
    res_file = bridge / "results" / f"res_{uid}.json"
    res_file.write_text(
        json.dumps(
            {
                "id": uid,
                "status": "filled",
                "ticket": 1,
                "price": 1.2345,
                "error": None,
                "sl": 1.2330,
                "tp": 1.2350,
            }
        ),
        encoding="utf-8",
    )

    res = br._wait_for_result(uid, 0.1)
    assert getattr(res, "sl", None) == pytest.approx(1.2330)
    assert getattr(res, "tp", None) == pytest.approx(1.2350)


def test_bad_stop_level_file_is_ignored(tmp_path: Path) -> None:
    bridge = tmp_path / "bridge"
    state = bridge / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "stop_level_EURUSD.json").write_text("{bad json", encoding="utf-8")

    br = MT4Broker(bridge_dir=bridge, symbol="EURUSD", timeout_sec=0.2)
    br.connect()
    br.market_order("BUY", 0.1, price=1.2345, sl=1.2330, tp=1.2360)

    cmd = next((bridge / "commands").glob("cmd_*.json"))
    data = json.loads(cmd.read_text(encoding="utf-8"))
    assert data["sl"] == 1.2330
    assert data["tp"] == 1.2360
