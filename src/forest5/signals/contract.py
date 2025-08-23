"""Data contract for technical trading signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# Action type for trading signals
Action = Literal["BUY", "SELL", "KEEP"]


@dataclass
class TechnicalSignal:
    """Contract describing a technical trading signal."""

    timeframe: str = ""
    action: Action = "KEEP"
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    horizon_minutes: int = 0
    technical_score: float = 0.0
    confidence_tech: float = 0.0
    drivers: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


__all__ = ["Action", "TechnicalSignal"]
