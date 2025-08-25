from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .contract import TechnicalSignal, Action
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
    engines can track which setup resulted in an opened position. Existing
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
class TriggeredSignal:
    """Signal emitted when a setup is triggered."""

    setup_id: str
    action: Action
    entry: float
    sl: float
    tp: float
    fill_price: float | None = None
    slippage: float = 0.0
    meta: dict[str, object] | None = None
    technical_score: float = 0.0
    confidence_tech: float = 1.0
    drivers: list[object] = field(default_factory=list)


class SetupRegistry:
    """Track armed trade setups and trigger on breakout."""

    def __init__(self, ttl_bars: int = 1) -> None:
        self.ttl_bars = ttl_bars
        self._setups: Dict[str, _ArmedSetup] = {}

    # ------------------------------------------------------------------
    def arm(
        self,
        key: str,
        index: int,
        signal: TechnicalSignal,
        *,
        ctx: TelemetryContext | None = None,
    ) -> str:
        """Store ``signal`` and arm it for the next bar.

        Returns
        -------
        setup_id:
            Identifier associated with the armed setup.
        """

        self._setups[key] = _ArmedSetup(signal=signal, expiry=index + self.ttl_bars, ctx=ctx)
        setup_id = getattr(signal, "id", key)
        if ctx is not None:
            log_event(
                E_SETUP_ARM,
                ctx=ctx,
                key=key,
                index=index,
                action=signal.action,
                setup_id=setup_id,
                entry=float(signal.entry),
                sl=float(signal.sl),
                tp=float(signal.tp),
            )
        return setup_id

    # ------------------------------------------------------------------
    def _check_price(
        self, *, index: int, price: float, ctx: TelemetryContext | None
    ) -> Optional[TriggeredSignal]:
        for key, setup in list(self._setups.items()):
            setup_ctx = setup.ctx or ctx

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
                fill_price = price
                slippage = (
                    fill_price - float(sig.entry)
                    if sig.action == "BUY"
                    else float(sig.entry) - fill_price
                )
                setup_id = getattr(sig, "id", key)
                if setup_ctx is not None:
                    log_event(
                        E_SETUP_TRIGGER,
                        ctx=setup_ctx,
                        trigger_price=price,
                        fill_price=fill_price,
                        slippage=slippage,
                        setup_id=setup_id,
                        action=sig.action,
                        entry=float(sig.entry),
                        sl=float(sig.sl),
                        tp=float(sig.tp),
                    )
                return TriggeredSignal(
                    setup_id=setup_id,
                    action=sig.action,
                    entry=float(sig.entry),
                    sl=float(sig.sl),
                    tp=float(sig.tp),
                    fill_price=fill_price,
                    slippage=slippage,
                    meta=getattr(sig, "meta", None),
                    technical_score=getattr(sig, "technical_score", 0.0),
                    confidence_tech=getattr(sig, "confidence_tech", 1.0),
                    drivers=list(getattr(sig, "drivers", [])),
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

    # ------------------------------------------------------------------
    def check(
        self,
        *,
        index: int,
        price: float | None = None,
        high: float | None = None,
        low: float | None = None,
        ctx: TelemetryContext | None = None,
    ) -> Optional[TriggeredSignal]:
        """Check armed setups for a trigger or expiry.

        For backward compatibility ``high``/``low`` may be supplied instead of
        ``price``. In that case ``high`` is evaluated first and ``low`` is only
        checked if ``high`` did not trigger a setup.
        """

        if price is not None:
            return self._check_price(index=index, price=price, ctx=ctx)

        triggered = None
        if high is not None:
            triggered = self._check_price(index=index, price=high, ctx=ctx)
        if triggered is None and low is not None:
            triggered = self._check_price(index=index, price=low, ctx=ctx)
        return triggered


__all__ = ["SetupRegistry", "SetupCandidate", "TriggeredSignal"]
