from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass

import structlog

# -- Event names -----------------------------------------------------------
# These constants identify high level telemetry events.  They intentionally
# use short string codes so that they are easy to search for in aggregated log
# output.

# Application setup and lifecycle.
E_SETUP_ARM = "setup_arm"
E_SETUP_TRIGGER = "setup_trigger"
E_SETUP_EXPIRE = "setup_expire"
E_SETUP_DONE = "setup_done"

# Trading related events.
E_ORDER_PLACED = "order_placed"
E_ORDER_FILLED = "order_filled"
E_ORDER_CANCELLED = "order_cancelled"

# Generic error/diagnostics events.
E_ERROR = "error"


# -- Reason codes ----------------------------------------------------------
# Reason codes provide additional colour for a given event.  As with the event
# names they are short and machine friendly.
R_TIMEONLY_WAIT = "timeonly_wait"
R_TIMEOUT = "timeout"
R_CANCELLED = "cancelled"
R_ERROR = "error"


@dataclass(slots=True)
class TelemetryContext:
    """Context information that will be bound to every telemetry event.

    The fields are intentionally optional so that callers can supply whatever
    identifiers they have available.
    """

    run_id: str | None = None
    strategy_id: str | None = None
    order_id: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    strategy: str | None = None
    setup_id: str | None = None


def new_id(prefix: str) -> str:
    """Return a short unique identifier with ``prefix``.

    Examples
    --------
    >>> new_id("run")  # doctest: +SKIP
    'run_4f9d2ab3'
    """

    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def log_event(event: str, ctx: TelemetryContext | None = None, **fields) -> None:
    """Log ``event`` via structlog, binding context and extra fields.

    Parameters
    ----------
    event:
        Event name constant, e.g. :data:`E_ORDER_FILLED`.
    ctx:
        Optional :class:`TelemetryContext` whose non-``None`` attributes will be
        bound to the log record.
    **fields:
        Additional key/value pairs describing the event.
    """

    logger = structlog.get_logger()
    if ctx is not None:
        # Only bind values that are not ``None`` to keep the log output compact.
        logger = logger.bind(**{k: v for k, v in asdict(ctx).items() if v is not None})
    logger = logger.bind(event=event)
    logger.info(event, **fields)


def setup_logger(level: str = "INFO"):
    """Configure and return a structlog logger.

    Parameters
    ----------
    level:
        Logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.

    Returns
    -------
    structlog.stdlib.BoundLogger
        Configured logger instance.
    """

    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper(), logging.INFO))
    processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ]
    structlog.configure(processors=processors)
    return structlog.get_logger()


# The module no longer exposes a global ``log`` instance. Call ``setup_logger``
# from application entry points to create one.
