from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class NumerologyRules:
    enabled: bool = False
    blocked_weekdays: list[int] = None  # 0..6
    blocked_hours: list[int] = None  # 0..23

    def __post_init__(self) -> None:
        self.blocked_weekdays = self.blocked_weekdays or []
        self.blocked_hours = self.blocked_hours or []


def is_trade_allowed(ts: datetime, rules: NumerologyRules) -> bool:
    if not rules.enabled:
        return True
    if ts.weekday() in rules.blocked_weekdays:
        return False
    if ts.hour in rules.blocked_hours:
        return False
    return True
