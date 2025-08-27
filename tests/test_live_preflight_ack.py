import json
import threading
import time
from forest5.cli import main


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


def test_live_preflight_ack(tmp_path, capsys):
    bridge = _prepare_bridge(tmp_path)

    def writer():
        cmd_dir = bridge / "commands"
        while True:
            reqs = list(cmd_dir.glob("req_*.json"))
            if reqs:
                uid = reqs[0].stem.split("_")[1]
                ack = bridge / "results" / f"ack_{uid}.json"
                ack.write_text(json.dumps(_sample_specs()), encoding="utf-8")
                break
            time.sleep(0.01)

    t = threading.Thread(target=writer)
    t.start()
    rc = main(
        [
            "live",
            "preflight",
            "--bridge-dir",
            str(bridge),
            "--symbol",
            "EURUSD",
            "--timeout",
            "1",
        ]
    )
    t.join()
    assert rc == 0
    ack_file = bridge / "handshake_ack.json"
    assert ack_file.exists()
    out = capsys.readouterr().out
    assert "preflight_ack" in out
