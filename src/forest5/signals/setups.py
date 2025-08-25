from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

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


@dataclass
class TriggeredSignal(TechnicalSignal):
    """Signal emitted when a setup is triggered.

    It extends :class:`TechnicalSignal` with execution details. Existing code
    that expects :class:`TechnicalSignal` continues to work.
    """

    key: str = ""
    trigger_price: float = 0.0
    fill_price: float = 0.0
    slippage: float = 0.0
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

    def check(
        self,
        *,
        index: int,
        price: float,
        ctx: TelemetryContext | None = None,
    ) -> Optional[TriggeredSignal]:
        """Check all armed setups for a trigger or expiry.

        Parameters
        ----------
        index:
            Index of the bar to evaluate.
        price:
            Current price used to detect breakouts.
        ctx:
            Optional context used when logging expiry events if the setup did
            not specify its own.
        """

        for key, setup in list(self._setups.items()):
            setup_ctx = setup.ctx or ctx

            # Expire old setups
            if index > setup.expiry:
                del self._setups[key]
                if setup_ctx is not None:
                    log_event(
                        E_SETUP_EXPIRE,
                        ctx=setup_ctx,
                        key=key,
                        index=index,
                        reason=R_TIMEOUT,
                    )
                continue

            sig = setup.signal
            triggered = (sig.action == "BUY" and price >= sig.entry) or (
                sig.action == "SELL" and price <= sig.entry
            )

            if triggered:
                del self._setups[key]
                trigger_price = price
                fill_price = price
                if sig.action == "BUY":
                    slippage = fill_price - float(sig.entry)
                else:
                    slippage = float(sig.entry) - fill_price
                if setup_ctx is not None:
                    log_event(
                        E_SETUP_TRIGGER,
                        ctx=setup_ctx,
                        trigger_price=trigger_price,
                        fill_price=fill_price,
                        slippage=slippage,
                        setup_id=getattr(sig, "id", None),
                        action=sig.action,
                        entry=float(sig.entry),
                        sl=float(sig.sl),
                        tp=float(sig.tp),
                    )
                return TriggeredSignal(
                    timeframe=sig.timeframe,
                    action=sig.action,
                    entry=sig.entry,
                    sl=sig.sl,
                    tp=sig.tp,
                    horizon_minutes=sig.horizon_minutes,
                    technical_score=sig.technical_score,
                    confidence_tech=sig.confidence_tech,
                    drivers=sig.drivers,
                    meta=sig.meta,
                    key=key,
                    trigger_price=trigger_price,
                    fill_price=fill_price,
                    slippage=slippage,
                    ctx=setup_ctx,
                )

            if index >= setup.expiry:
                del self._setups[key]
                if setup_ctx is not None:
                    log_event(
                        E_SETUP_EXPIRE,
                        ctx=setup_ctx,
                        key=key,
                        index=index,
                        reason=R_TIMEOUT,
                    )

        return None


__all__ = ["SetupRegistry", "SetupCandidate", "TriggeredSignal"]
