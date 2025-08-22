from __future__ import annotations

import logging
import structlog


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

    logging.basicConfig(
        format="%(message)s", level=getattr(logging, level.upper(), logging.INFO)
    )
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
