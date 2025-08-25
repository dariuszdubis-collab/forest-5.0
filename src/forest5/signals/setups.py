from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .contract import TechnicalSignal
from forest5.utils.log import (
    E_SETUP_ARM,
    E_SETUP_EXPIRE,
    E_SETUP_TRIGGER,
    R_TIMEOUT,
    TelemetryContext,
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
        *,
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

        self._setups[key] = _ArmedSetup(signal=signal, expiry=index + self.ttl_bars, ctx=ctx)
        if ctx is not None:
            log_event(
                E_SETUP_ARM,
                ctx=ctx,
                key=key,
                index=index,
                action=signal.action,
                setup_id=getattr(signal, "id", None),
                entry=float(signal.entry),
                sl=float(signal.sl),
                tp=float(signal.tp),
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
            if setup.ctx is not None:
                log_event(
                    E_SETUP_EXPIRE,
                    ctx=setup.ctx,
                    key=key,
                    index=index,
                    reason=R_TIMEOUT,
                )
            return None

        sig = setup.signal
        triggered = (sig.action == "BUY" and high >= sig.entry) or (
            sig.action == "SELL" and low <= sig.entry
        )

        if triggered:
            del self._setups[key]
            trigger_price = high if sig.action == "BUY" else low
            fill_price = trigger_price
            if sig.action == "BUY":
                slippage = fill_price - float(sig.entry)
            else:
                slippage = float(sig.entry) - fill_price
            if setup.ctx is not None:
                log_event(
                    E_SETUP_TRIGGER,
                    ctx=setup.ctx,
                    trigger_price=trigger_price,
                    fill_price=fill_price,
                    slippage=slippage,
                    setup_id=getattr(sig, "id", None),
                    action=sig.action,
                    entry=float(sig.entry),
                    sl=float(sig.sl),
                    tp=float(sig.tp),
                )
            return sig

        if index >= setup.expiry:
            del self._setups[key]
            if setup.ctx is not None:
                log_event(
                    E_SETUP_EXPIRE,
                    ctx=setup.ctx,
                    key=key,
                    index=index,
                    reason=R_TIMEOUT,
                )
        return None


__all__ = ["SetupRegistry", "SetupCandidate"]
