import json, threading, time, importlib
from pathlib import Path

# Spróbuj odnaleźć MT4Broker niezależnie od dokładnej ścieżki modułu
MT4Broker = None
for name in ("forest5.broker.mt4_broker", "forest5.live.mt4_broker", "mt4_broker"):
    try:
        mod = importlib.import_module(name)
        MT4Broker = getattr(mod, "MT4Broker")
        break
    except Exception:
        pass
if MT4Broker is None:
    raise SystemExit("MT4Broker module not found")

def fake_ea_loop(bridge: Path, stop):
    cmd = bridge/"commands"; res = bridge/"results"
    (bridge/"ticks").mkdir(parents=True, exist_ok=True)
    (bridge/"state").mkdir(parents=True, exist_ok=True)
    (bridge/"ticks"/"tick.json").write_text('{"symbol":"EURUSD","bid":1.0,"ask":1.0,"time":0}', encoding="utf-8")
    (bridge/"state"/"account.json").write_text('{"equity":10000}', encoding="utf-8")
    (bridge/"state"/"position_EURUSD.json").write_text('{"qty":0}', encoding="utf-8")
    cmd.mkdir(parents=True, exist_ok=True); res.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        for p in cmd.glob("cmd_*.json"):
            s = json.loads(p.read_text(encoding="utf-8"))
            rid = p.stem[4:]
            (res / f"res_{rid}.json").write_text(
                json.dumps(
                    {
                        "id": rid,
                        "status": "filled",
                        "ticket": 1,
                        "price": 1.2345,
                        "error": None,
                    }
                ),
                encoding="utf-8",
            )
            p.unlink(missing_ok=True)
        time.sleep(0.05)

def test_broker_file_bridge(tmp_path: Path):
    bridge = tmp_path/"forest_bridge"
    stop = threading.Event()
    t = threading.Thread(target=fake_ea_loop, args=(bridge,stop), daemon=True); t.start()
    try:
        br = MT4Broker(bridge_dir=bridge, symbol="EURUSD", timeout_sec=2.0)
        br.connect()
        r = br.market_order("BUY", 0.01)
        assert r["status"] == "filled"
        assert br.equity() == 10000
        assert br.position_qty() == 0
    finally:
        stop.set(); t.join(timeout=1)
