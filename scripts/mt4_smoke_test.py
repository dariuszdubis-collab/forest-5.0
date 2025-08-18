#!/usr/bin/env python3
import os, json, time, uuid, sys, pathlib
BRIDGE = pathlib.Path(os.environ.get("FOREST_MT4_BRIDGE_DIR","" )).expanduser()
if not BRIDGE:
    print("Set FOREST_MT4_BRIDGE_DIR to .../MQL4/Files/forest_bridge", file=sys.stderr); sys.exit(2)
cmd = BRIDGE/"commands"; res = BRIDGE/"results"
cmd.mkdir(parents=True, exist_ok=True); res.mkdir(parents=True, exist_ok=True)
cid = uuid.uuid4().hex
cmdp = cmd/f"cmd_{cid}.json"; resp = res/f"res_{cid}.json"
with cmdp.open("w") as f:
    json.dump({"id":cid,"action":"BUY","symbol":"EURUSD","volume":0.01,"sl":None,"tp":None}, f)
print("Sent:", cmdp)
deadline = time.time()+10
while time.time()<deadline:
    if resp.exists():
        print("Result:", resp.read_text()); sys.exit(0)
    time.sleep(0.2)
print("No EA response. Check MT4 -> Experts/Journal.", file=sys.stderr); sys.exit(1)
