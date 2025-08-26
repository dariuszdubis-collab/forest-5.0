import json
import logging
import sys
from pathlib import Path

from datetime import datetime, timedelta

import pytest
import structlog

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from forest5.utils.log import (  # noqa: E402
    TelemetryContext,
    E_SETUP_TRIGGER,
    E_SETUP_EXPIRE,
    R_TIMEOUT,
)
from forest5.signals.setups import SetupRegistry, SetupCandidate  # noqa: E402


def test_setup_trigger_and_expire_logging(caplog):
    structlog.reset_defaults()
    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        processors=[structlog.processors.JSONRenderer()],
    )
    caplog.set_level(logging.INFO)

    reg = SetupRegistry()

    ctx = TelemetryContext(run_id="run1", symbol="EURUSD", timeframe="H1", setup_id="s1")
    sig = SetupCandidate(action="BUY", entry=1.0, sl=0.9, tp=1.1, id="s1")
    t0 = datetime(2024, 1, 1, 0, 0)
    reg.arm("EURUSD", 0, sig, bar_time=t0, ctx=ctx)
    reg.check(index=0, price=1.2, now=t0)

    ctx2 = TelemetryContext(run_id="run1", symbol="EURUSD", timeframe="H1", setup_id="s2")
    sig2 = SetupCandidate(action="BUY", entry=1.0, sl=0.9, tp=1.1, id="s2")
    t1 = t0 + timedelta(minutes=1)
    reg.arm("EURUSD", 1, sig2, bar_time=t1, ctx=ctx2)
    t2 = t0 + timedelta(minutes=2)
    reg.check(index=2, price=0.95, now=t2)

    records = [json.loads(r.message) for r in caplog.records]

    trig = next(r for r in records if r["event"] == E_SETUP_TRIGGER)
    assert trig["event"] == E_SETUP_TRIGGER
    assert trig["trigger_price"] == 1.2
    assert trig["fill_price"] == 1.2
    assert trig["slippage"] == pytest.approx(0.2)
    assert trig["setup_id"] == "s1"
    assert trig["action"] == "BUY"
    assert trig["entry"] == 1.0
    assert trig["sl"] == 0.9
    assert trig["tp"] == 1.1

    exp = next(r for r in records if r["event"] == E_SETUP_EXPIRE)
    assert exp["event"] == E_SETUP_EXPIRE
    assert exp["reason"] == R_TIMEOUT

    structlog.reset_defaults()
