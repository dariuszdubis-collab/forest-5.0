from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .contract import TechnicalSignal


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

    def arm(self, key: str, index: int, signal: TechnicalSignal) -> None:
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

        self._setups[key] = _ArmedSetup(signal=signal, expiry=index + self.ttl_bars)

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
            return None

        sig = setup.signal
        triggered = (sig.action == "BUY" and high >= sig.entry) or (
            sig.action == "SELL" and low <= sig.entry
        )

        if triggered:
            del self._setups[key]
            return sig

        if index >= setup.expiry:
            del self._setups[key]
        return None


__all__ = ["SetupRegistry", "SetupCandidate"]
