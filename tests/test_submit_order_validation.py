import json
import logging
import structlog

from forest5.live.router import PaperBroker, submit_order
from forest5.utils.log import (
    E_ORDER_ACK,
    E_ORDER_FILLED,
    E_ORDER_REJECTED,
    E_ORDER_SUBMITTED,
    TelemetryContext,
    setup_logger,
)


def _configure_logging():
    processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ]
    setup_logger()
    structlog.configure(processors=processors, logger_factory=structlog.stdlib.LoggerFactory())
    return processors


def test_submit_order_invalid_qty(caplog):
    processors = _configure_logging()
    broker = PaperBroker()
    broker.connect()
    ctx = TelemetryContext(run_id="test_run", symbol="EURUSD")

    with caplog.at_level(logging.INFO):
        result = submit_order(broker, "BUY", 0.0, price=1.0, ctx=ctx)

    records = [json.loads(r.message) for r in caplog.records]
    assert [r["event"] for r in records] == [E_ORDER_REJECTED]
    assert records[0]["reason"] == "invalid_qty"
    assert result.status == "rejected"
    structlog.configure(processors=processors)


def test_submit_order_invalid_stops(caplog):
    processors = _configure_logging()
    broker = PaperBroker()
    broker.connect()
    ctx = TelemetryContext(run_id="test_run", symbol="EURUSD")

    with caplog.at_level(logging.INFO):
        result = submit_order(broker, "BUY", 1.0, price=1.0, sl=float("nan"), ctx=ctx)

    records = [json.loads(r.message) for r in caplog.records]
    assert [r["event"] for r in records] == [E_ORDER_REJECTED]
    assert records[0]["reason"] == "invalid_stops"
    assert result.status == "rejected"
    structlog.configure(processors=processors)


def test_submit_order_auto_client_order_id(caplog):
    processors = _configure_logging()
    broker = PaperBroker()
    broker.connect()
    ctx = TelemetryContext(run_id="test_run", symbol="EURUSD")

    with caplog.at_level(logging.INFO):
        submit_order(broker, "BUY", 1.0, price=1.2345, ctx=ctx)

    records = [json.loads(r.message) for r in caplog.records]
    events = [r["event"] for r in records]
    assert events == [E_ORDER_SUBMITTED, E_ORDER_ACK, E_ORDER_FILLED]
    cids = {r.get("client_order_id") for r in records}
    assert len(cids) == 1
    assert cids.pop()
    structlog.configure(processors=processors)
