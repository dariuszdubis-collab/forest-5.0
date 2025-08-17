from __future__ import annotations

import logging
from typing import Iterable

import structlog
from structlog.typing import Processor


def setup_logger(level: str = "INFO"):
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper(), logging.INFO))
    processors: Iterable[Processor] = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ]
    structlog.configure(processors=processors)
    return structlog.get_logger()


log = setup_logger()
