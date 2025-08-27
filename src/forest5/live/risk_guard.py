"""Risk guard utilities and state tracking.

This module provides a small helper that tracks the activation state of
various risk limits and emits structured log events when those limits are
crossed.  Only a minimal drawdown guard is implemented but the interface is
generic so more checks can be added in the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict

from ..utils.log import (
    E_RISK_GUARD_ACTIVE,
    E_RISK_GUARD_CLEARED,
    log_event,
)


@dataclass
class RiskGuard:
    """Track whether individual risk limits are active."""

    _active: Dict[str, bool] = field(default_factory=dict)

    def _set_active(self, guard: str, active: bool, **fields) -> None:
        """Emit a state transition event for ``guard`` if needed.

        Only changes of state trigger an event which keeps log noise to a
        minimum.  ``fields`` are included in the telemetry event.
        """

        prev = self._active.get(guard, False)
        if active and not prev:
            log_event(E_RISK_GUARD_ACTIVE, guard=guard, **fields)
        elif not active and prev:
            log_event(E_RISK_GUARD_CLEARED, guard=guard, **fields)
        self._active[guard] = active

    # ------------------------------------------------------------------
    # Drawdown guard
    # ------------------------------------------------------------------
    def should_halt_for_drawdown(self, start_eq: float, cur_eq: float, max_dd: float) -> bool:
        """Return ``True`` if drawdown from ``start_eq`` to ``cur_eq`` meets ``max_dd``.

        ``max_dd`` is expressed as a fraction (e.g. ``0.20`` for 20%).  Whenever
        the threshold is crossed the appropriate risk‑guard event is logged.  The
        event is emitted only once per state change.
        """

        if start_eq <= 0 or max_dd <= 0:
            self._set_active("max_daily_loss", False, day=date.today().isoformat())
            return False
        dd = (start_eq - cur_eq) / start_eq
        day_str = date.today().isoformat()
        if dd >= max_dd:
            self._set_active(
                "max_daily_loss",
                True,
                value=dd,
                limit=max_dd,
                day=day_str,
                reason="threshold_exceeded",
            )
            return True
        self._set_active("max_daily_loss", False, day=day_str, reason="threshold_exceeded")
        return False


# Backwards compatible helper -------------------------------------------------
_GLOBAL_GUARD = RiskGuard()


def should_halt_for_drawdown(start_eq: float, cur_eq: float, max_dd: float) -> bool:
    """Backward compatible wrapper around :class:`RiskGuard`.

    Existing code expects a simple function.  We delegate to a global
    :class:`RiskGuard` instance which also ensures state is remembered across
    calls during a session.
    """

    return _GLOBAL_GUARD.should_halt_for_drawdown(start_eq, cur_eq, max_dd)


__all__ = [
    "RiskGuard",
    "should_halt_for_drawdown",
]
