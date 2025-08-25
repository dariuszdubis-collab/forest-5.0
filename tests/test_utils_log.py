from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import structlog

spec = spec_from_file_location(
    "log", Path(__file__).resolve().parents[1] / "src/forest5/utils/log.py"
)
log = module_from_spec(spec)
sys.modules[spec.name] = log
spec.loader.exec_module(log)  # type: ignore[misc]

E_ORDER_PLACED = log.E_ORDER_PLACED
TelemetryContext = log.TelemetryContext
log_event = log.log_event


class DummyLogger:
    def __init__(self):
        self.bound = {}
        self.records = []

    def bind(self, **kwargs):
        self.bound.update(kwargs)
        return self

    def info(self, *args, **fields):
        event = args[0] if args else fields.pop("event")
        fields.pop("event", None)
        record = {"event": event, **self.bound, **fields}
        self.records.append(record)


def test_log_event_binds_only_non_none_fields():
    logger = DummyLogger()
    orig_get_logger = structlog.get_logger
    structlog.get_logger = lambda: logger
    try:
        ctx = TelemetryContext(
            run_id="run123",
            strategy_id=None,
            order_id="order456",
            symbol="EURUSD",
            timeframe=None,
            strategy="rsi",
        )
        log_event(E_ORDER_PLACED, ctx, extra="foo")
    finally:
        structlog.get_logger = orig_get_logger
    event = logger.records[0]
    assert event["run_id"] == "run123"
    assert "strategy_id" not in event
    assert event["order_id"] == "order456"
    assert event["symbol"] == "EURUSD"
    assert "timeframe" not in event
    assert event["strategy"] == "rsi"
    assert event["extra"] == "foo"
    assert event["event"] == E_ORDER_PLACED
