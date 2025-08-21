#!/usr/bin/env python3

"""Minimal smoke test for the MT4 bridge.

The script drops a single BUY command into the bridge's ``commands`` folder and
waits for a corresponding response file to appear in ``results``.  It is a
lightweight check to verify that the Expert Advisor can communicate with this
Python environment.
"""

import json
import os
import pathlib
import sys
import time
import uuid


BRIDGE = pathlib.Path(os.environ.get("FOREST_MT4_BRIDGE_DIR", "")).expanduser()

if not BRIDGE:
    print(
        "Set FOREST_MT4_BRIDGE_DIR to .../MQL4/Files/forest_bridge",
        file=sys.stderr,
    )
    sys.exit(2)

cmd_dir = BRIDGE / "commands"
res_dir = BRIDGE / "results"
cmd_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

cid = uuid.uuid4().hex
cmd_path = cmd_dir / f"cmd_{cid}.json"
res_path = res_dir / f"res_{cid}.json"

with cmd_path.open("w") as f:
    json.dump(
        {
            "id": cid,
            "action": "BUY",
            "symbol": "EURUSD",
            "volume": 0.01,
            "sl": None,
            "tp": None,
        },
        f,
    )

print("Sent:", cmd_path)
deadline = time.time() + 10
while time.time() < deadline:
    if res_path.exists():
        print("Result:", res_path.read_text())
        sys.exit(0)
    time.sleep(0.2)

print("No EA response. Check MT4 -> Experts/Journal.", file=sys.stderr)
sys.exit(1)
