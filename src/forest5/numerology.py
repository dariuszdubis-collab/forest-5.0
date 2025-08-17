from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class NumerologyRules:
    enabled: bool = False
    blocked_weekdays: list[int] = field(default_factory=list)  # 0..6
    blocked_hours: list[int] = field(default_factory=list)  # 0..23


def is_trade_allowed(ts: datetime, rules: NumerologyRules) -> bool:
    if not rules.enabled:
        return True
    if ts.weekday() in rules.blocked_weekdays:
        return False
    if ts.hour in rules.blocked_hours:
        return False
    return True
