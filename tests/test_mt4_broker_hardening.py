import json, threading, time, importlib
from pathlib import Path
import pytest

def _import_mt4broker():
    """
    Szukamy klasy MT4Broker w kilku możliwych lokalizacjach.
    Łapiemy TYLKO ModuleNotFoundError – inne wyjątki nie są
    maskowane, żeby nie ukrywać realnych błędów modułu.
    """
    candidates = (
        "forest5.broker.mt4_broker",
        "forest5.live.mt4_broker",
        "mt4_broker",
        "src.forest5.live.mt4_broker",
    )
    last_mnf = []
    for name in candidates:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError as e:
            last_mnf.append(f"{name}: {e}")
            continue
        if hasattr(mod, "MT4Broker"):
            return getattr(mod, "MT4Broker")
    raise AssertionError("MT4Broker not found. Tried: " + ", ".join(candidates) +
                         (" | import errors: " + " ; ".join(last_mnf) if last_mnf else ""))

MT4Broker = _import_mt4broker()

def _write_text(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def _fake_ea_loop(bridge: Path, stop_evt: threading.Event):
    cmd = bridge / "commands"
    res = bridge / "results"
    ticks = bridge / "ticks"
    state = bridge / "state"

    _write_text(ticks / "tick.json", '{"symbol":"EURUSD","bid":1.0,"ask":1.0,"time":0}')
    _write_text(state / "account.json", '{"equity":10000}')
    _write_text(state / "position_EURUSD.json", '{"qty":0}')

    cmd.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    while not stop_evt.is_set():
        for p in list(cmd.glob("cmd_*.json")):
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
                rid = p.stem[4:]  # po 'cmd_'
                _write_text(res / f"res_{rid}.json",
                            json.dumps({"id": rid, "status": "filled",
                                        "ticket": 1, "price": 1.23456, "error": None}))
                p.unlink(missing_ok=True)
            except Exception:
                # Nie maskujemy błędów modułów – ale w testowej pętli EA
                # dopuszczamy bezpieczne pominięcie pojedynczego pliku.
                pass
        time.sleep(0.02)

def test_timeout_returns_result_object(tmp_path: Path):
    """
    Repo-umowa: przy braku odpowiedzi EA broker NIE rzuca TimeoutError,
    tylko zwraca obiekt wyniku z informacją 'timeout' (w statusie lub error).
    """
    bridge = tmp_path / "forest_bridge"
    # Minimalna struktura, ale BEZ pętli EA -> zasymuluj timeout
    for d in ("ticks", "commands", "results", "state"):
        (bridge / d).mkdir(parents=True, exist_ok=True)
    _write_text(bridge / "ticks" / "tick.json",
                '{"symbol":"EURUSD","bid":1,"ask":1,"time":0}')

    br = MT4Broker(bridge_dir=bridge, symbol="EURUSD", timeout_sec=0.5)
    br.connect()
    result = br.market_order("BUY", 0.01)

    assert result["status"].lower() == "rejected"
    assert str(result["error"]).lower() == "timeout"

def test_back_to_back_orders_and_cleanup(tmp_path: Path):
    bridge = tmp_path / "forest_bridge"
    stop = threading.Event()
    t = threading.Thread(target=_fake_ea_loop, args=(bridge, stop), daemon=True)
    t.start()
    try:
        br = MT4Broker(bridge_dir=bridge, symbol="EURUSD", timeout_sec=2.0)
        br.connect()
        r1 = br.market_order("BUY", 0.01)
        r2 = br.market_order("SELL", 0.01)
        assert r1["status"] == "filled"
        assert r2["status"] == "filled"
        # komendy muszą być posprzątane przez EA
        assert not list((bridge / "commands").glob("cmd_*.json"))
        # sanity state api
        assert isinstance(br.equity(), float)
        assert isinstance(br.position_qty(), float)
    finally:
        stop.set()
        t.join(timeout=1)
