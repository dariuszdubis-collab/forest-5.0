from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TechnicalSignal:
    """Contract describing a technical trading signal."""

    action: int = 0
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    technical_score: float = 0.0
    confidence_tech: float = 0.0
