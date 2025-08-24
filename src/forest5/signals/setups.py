from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .contract import TechnicalSignal
from ..utils.log import (
    TelemetryContext,
    E_SETUP_ARM,
    E_SETUP_EXPIRE,
    E_SETUP_TRIGGER,
    R_TTL,
    log_event,
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
class _ArmedSetup:
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
        ctx: TelemetryContext | None = None,
    ) -> None:
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

        self._setups[key] = _ArmedSetup(
            signal=signal, expiry=index + self.ttl_bars, ctx=ctx
        )
        log_event(
            E_SETUP_ARM,
            ctx,
            action=signal.action,
            entry=signal.entry,
            sl=signal.sl,
            tp=signal.tp,
            pattern=signal.meta.get("pattern"),
            drivers=signal.drivers,
            ttl_bars=self.ttl_bars,
        )

    def check(self, key: str, index: int, high: float, low: float) -> TechnicalSignal | None:
        """Check for triggered or expired setups.

        Parameters
        ----------
        key:
            Identifier used when arming the setup.
        index:
            Index of the bar to evaluate.
        high, low:
            High and low price of the current bar used to detect breakouts.
        """

        setup = self._setups.get(key)
        if setup is None:
            return None

        # Expire old setups
        if index > setup.expiry:
            del self._setups[key]
            log_event(
                E_SETUP_EXPIRE,
                setup.ctx,
                action=setup.signal.action,
                entry=setup.signal.entry,
                sl=setup.signal.sl,
                tp=setup.signal.tp,
                pattern=setup.signal.meta.get("pattern"),
                drivers=setup.signal.drivers,
                reason=R_TTL,
            )
            return None

        sig = setup.signal
        triggered = (sig.action == "BUY" and high >= sig.entry) or (
            sig.action == "SELL" and low <= sig.entry
        )

        if triggered:
            del self._setups[key]
            log_event(
                E_SETUP_TRIGGER,
                setup.ctx,
                action=sig.action,
                entry=sig.entry,
                sl=sig.sl,
                tp=sig.tp,
                pattern=sig.meta.get("pattern"),
                drivers=sig.drivers,
            )
            return sig

        if index >= setup.expiry:
            del self._setups[key]
            log_event(
                E_SETUP_EXPIRE,
                setup.ctx,
                action=sig.action,
                entry=sig.entry,
                sl=sig.sl,
                tp=sig.tp,
                pattern=sig.meta.get("pattern"),
                drivers=sig.drivers,
                reason=R_TTL,
            )
        return None


__all__ = ["SetupRegistry", "SetupCandidate"]
