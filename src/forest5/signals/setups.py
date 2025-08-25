from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict

from .contract import TechnicalSignal
from forest5.utils.log import (
    E_SETUP_ARM,
    E_SETUP_TRIGGER,
    E_SETUP_EXPIRE,
    TelemetryContext,
    log_event,
    new_id,
)


@dataclass
class SetupCandidate(TechnicalSignal):
    """Candidate trade setup derived from :class:`TechnicalSignal`.

    It extends :class:`TechnicalSignal` with an identifier so that backtest
    engines can track which setup resulted in an opened position.  Existing
    code that expects a :class:`TechnicalSignal` continues to work because
    :class:`SetupCandidate` subclasses it.
    """

    id: str = ""


@dataclass
class TriggeredSignal(TechnicalSignal):
    """Signal emitted when an armed setup triggers."""

    setup_id: str = ""


@dataclass
class _ArmedSetup:
    key: str
    signal: TechnicalSignal
    expiry: int
    ctx: TelemetryContext | None = None


class SetupRegistry:
    """Track armed trade setups and trigger on breakout.

    Setups are stored with a time-to-live (TTL) measured in bars. Each armed
    setup is valid for exactly one subsequent bar. If a breakout beyond the
    entry price occurs within that bar the stored :class:`TechnicalSignal` is
    emitted. Otherwise the setup is discarded after the bar completes.
    """

    def __init__(self, ttl_bars: int = 1) -> None:
        self.ttl_bars = ttl_bars
        self._setups: Dict[str, _ArmedSetup] = {}

    def arm(
        self,
        key: str,
        index: int,
        signal: TechnicalSignal,
        *,
        ctx: TelemetryContext | None = None,
    ) -> str:
        """Store ``signal`` and arm it for the next bar.

        Parameters
        ----------
        key:
            Identifier for the setup (e.g. symbol or timeframe).
        index:
            Index of the current bar.  The setup will expire after
            ``index + ttl_bars``.
        signal:
            Fully populated :class:`TechnicalSignal` describing the trade to
            execute upon breakout.
        """

        setup_id = new_id("setup")
        self._setups[setup_id] = _ArmedSetup(
            key=key, signal=signal, expiry=index + self.ttl_bars, ctx=ctx
        )
        if ctx is not None:
            log_event(
                E_SETUP_ARM,
                ctx=ctx,
                key=key,
                index=index,
                action=signal.action,
                setup_id=setup_id,
            )
        return setup_id

    def check(
        self,
        index: int,
        price: float,
        *,
        ctx: TelemetryContext | None = None,
    ) -> TriggeredSignal | None:
        """Check for triggered or expired setups at ``price``.

        Parameters
        ----------
        index:
            Index of the bar to evaluate.
        price:
            Current price used to detect breakouts.
        """

        for setup_id, setup in list(self._setups.items()):
            if index > setup.expiry:
                del self._setups[setup_id]
                log_event(
                    E_SETUP_EXPIRE,
                    ctx=setup.ctx or ctx,
                    key=setup.key,
                    index=index,
                    setup_id=setup_id,
                )
                continue

            sig = setup.signal
            triggered = (sig.action == "BUY" and price >= sig.entry) or (
                sig.action == "SELL" and price <= sig.entry
            )
            if triggered:
                del self._setups[setup_id]
                sig_data = {f.name: getattr(sig, f.name) for f in fields(TechnicalSignal)}
                res = TriggeredSignal(setup_id=setup_id, **sig_data)
                log_event(
                    E_SETUP_TRIGGER,
                    ctx=setup.ctx or ctx,
                    key=setup.key,
                    index=index,
                    action=sig.action,
                    setup_id=setup_id,
                )
                return res

        return None


__all__ = ["SetupRegistry", "SetupCandidate", "TriggeredSignal"]
