import json
import time

from forest5.live.mt4_broker import MT4Broker


def _prepare_bridge(tmp_path):
    for d in ["commands", "results", "state", "ticks"]:
        (tmp_path / d).mkdir()
    return tmp_path


def _sample_specs():
    return {
        "digits": 5,
        "point": 0.00001,
        "tick_size": 0.00001,
        "min_lot": 0.01,
        "lot_step": 0.01,
        "contract_size": 100000,
        "stop_level": 10,
        "freeze_level": 0,
    }


def test_request_and_ack(tmp_path):
    bridge = _prepare_bridge(tmp_path)
    broker = MT4Broker(bridge_dir=bridge, symbol="EURUSD", timeout_sec=1.0)
    broker.connect()
    uid = broker.request_specs()

    ack_path = bridge / "results" / f"ack_{uid}.json"
    ack_path.write_text(json.dumps(_sample_specs()), encoding="utf-8")

    specs = broker.await_ack(uid, timeout=0.5)
    assert specs["digits"] == 5


def test_await_ack_timeout(tmp_path):
    bridge = _prepare_bridge(tmp_path)
    broker = MT4Broker(bridge_dir=bridge, symbol="EURUSD", timeout_sec=0.2)
    broker.connect()
    uid = broker.request_specs()

    start = time.time()
    try:
        broker.await_ack(uid, timeout=0.2)
        assert False, "expected timeout"
    except TimeoutError:
        pass
    assert time.time() - start >= 0.2
