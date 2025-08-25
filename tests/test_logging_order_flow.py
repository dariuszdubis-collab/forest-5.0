import json
import logging

import structlog

from forest5.live.router import PaperBroker, submit_order
from forest5.utils.log import (
    E_ORDER_ACK,
    E_ORDER_FILLED,
    E_ORDER_SUBMITTED,
    TelemetryContext,
    new_id,
    setup_logger,
)


def test_order_flow_logging(caplog):
    processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ]
    setup_logger()
    structlog.configure(processors=processors, logger_factory=structlog.stdlib.LoggerFactory())
    broker = PaperBroker()
    broker.connect()
    ctx = TelemetryContext(run_id="test_run", symbol="EURUSD")
    cid = new_id("cl")

    with caplog.at_level(logging.INFO):
        submit_order(broker, "BUY", 1.0, price=1.2345, ctx=ctx, client_order_id=cid)

    records = [json.loads(r.message) for r in caplog.records]
    assert [r["event"] for r in records] == [
        E_ORDER_SUBMITTED,
        E_ORDER_ACK,
        E_ORDER_FILLED,
    ]
    for r in records:
        assert r.get("client_order_id") == cid
    # restore default print logger for other tests
    structlog.configure(processors=processors)
