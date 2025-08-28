import json
import threading
import time

from forest5.cli import main

from test_live_preflight import _prepare_bridge, _sample_specs


def test_preflight_ack_passes(tmp_path, capsys):
    bridge = _prepare_bridge(tmp_path)

    def writer():
        ping = bridge / "PING"
        while not ping.exists():
            time.sleep(0.01)
        (bridge / "ACK").write_text("ACK", encoding="utf-8")
        cmd_dir = bridge / "commands"
        while True:
            reqs = list(cmd_dir.glob("req_*.json"))
            if reqs:
                uid = reqs[0].stem.split("_")[1]
                ack = bridge / "results" / f"ack_{uid}.json"
                ack.write_text(json.dumps(_sample_specs()), encoding="utf-8")
                break
            time.sleep(0.01)

    t = threading.Thread(target=writer, daemon=True)
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
    t.join(timeout=5)
    assert not t.is_alive(), "writer thread did not finish"
    assert rc == 0
    ack_path = bridge / "handshake_ack.json"
    assert ack_path.exists()
    data = json.loads(ack_path.read_text())
    assert data["symbol"] == "EURUSD"
    out = capsys.readouterr().out
    assert "preflight_ack" in out


def test_preflight_nack_times_out_and_fails(tmp_path, capsys):
    bridge = _prepare_bridge(tmp_path)
    rc = main(
        [
            "live",
            "preflight",
            "--bridge-dir",
            str(bridge),
            "--symbol",
            "EURUSD",
            "--timeout",
            "0.2",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "ACK" in err


def test_preflight_ignores_stale_ack(tmp_path, capsys):
    bridge = _prepare_bridge(tmp_path)
    # stale ACK left by previous run should not short-circuit handshake
    (bridge / "ACK").write_text("ACK", encoding="utf-8")
    rc = main(
        [
            "live",
            "preflight",
            "--bridge-dir",
            str(bridge),
            "--symbol",
            "EURUSD",
            "--timeout",
            "0.2",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "ACK" in err
