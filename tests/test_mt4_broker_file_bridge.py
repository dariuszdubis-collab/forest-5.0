import json, time, sys, types, importlib.util
from pathlib import Path
from threading import Thread

# Manually load MT4Broker without importing the whole forest5 package
ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "src" / "forest5" / "live"

forest5_pkg = types.ModuleType("forest5")
forest5_pkg.__path__ = [str(ROOT / "src" / "forest5")]
sys.modules.setdefault("forest5", forest5_pkg)
live_pkg = types.ModuleType("forest5.live")
live_pkg.__path__ = [str(LIVE_DIR)]
sys.modules.setdefault("forest5.live", live_pkg)

spec_router = importlib.util.spec_from_file_location(
    "forest5.live.router", LIVE_DIR / "router.py"
)
router_module = importlib.util.module_from_spec(spec_router)
assert spec_router.loader is not None
sys.modules["forest5.live.router"] = router_module
spec_router.loader.exec_module(router_module)

spec = importlib.util.spec_from_file_location(
    "forest5.live.mt4_broker", LIVE_DIR / "mt4_broker.py"
)
mt4_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["forest5.live.mt4_broker"] = mt4_module
spec.loader.exec_module(mt4_module)
MT4Broker = mt4_module.MT4Broker


def _respond_once(bridge: Path):
    """Wait for a single command file and write corresponding result."""
    cmd_dir = bridge / "commands"
    res_dir = bridge / "results"
    # Wait until a command file appears
    while True:
        cmds = list(cmd_dir.glob("cmd_*.json"))
        if cmds:
            p = cmds[0]
            data = json.loads(p.read_text())
            rid = data["id"]
            result = {"id": rid, "status": "filled", "ticket": 1, "price": 1.2345}
            (res_dir / f"res_{rid}.json").write_text(json.dumps(result))
            break
        time.sleep(0.05)


def test_market_order_success(tmp_path: Path):
    bridge = tmp_path / "bridge"
    br = MT4Broker(bridge_dir=bridge, symbol="EURUSD", timeout_sec=1.0)
    br.connect()
    # prepare state files for equity and position
    (bridge / "state" / "account.json").write_text("{\"equity\":1234}")
    (bridge / "state" / "position_EURUSD.json").write_text("{\"qty\":1.5}")
    # start helper thread to respond to command
    t = Thread(target=_respond_once, args=(bridge,), daemon=True)
    t.start()
    result = br.market_order("BUY", 0.1)
    t.join(timeout=1)
    assert result["status"] == "filled"
    assert br.equity() == 1234
    assert br.position_qty() == 1.5


def test_market_order_timeout(tmp_path: Path):
    bridge = tmp_path / "bridge"
    br = MT4Broker(bridge_dir=bridge, symbol="EURUSD", timeout_sec=0.2)
    br.connect()
    result = br.market_order("BUY", 0.1)
    assert result["status"] == "rejected"
    assert result["error"] == "timeout"


def test_position_and_equity(tmp_path: Path):
    bridge = tmp_path / "bridge"
    br = MT4Broker(bridge_dir=bridge, symbol="EURUSD")
    br.connect()
    (bridge / "state" / "account.json").write_text("{\"equity\":999}")
    (bridge / "state" / "position_EURUSD.json").write_text("{\"qty\":2.0}")
    assert br.equity() == 999
    assert br.position_qty() == 2.0
