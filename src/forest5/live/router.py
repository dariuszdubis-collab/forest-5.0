from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol
import math
import uuid

from ..utils.log import (
    E_ORDER_ACK,
    E_ORDER_FILLED,
    E_ORDER_REJECTED,
    E_ORDER_SUBMITTED,
    TelemetryContext,
    log_event,
)


class OrderRouter(Protocol):
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def set_price(self, price: float) -> None: ...
    def market_order(
        self,
        side: str,
        qty: float,
        price: Optional[float] = None,
        *,
        entry: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        ctx: TelemetryContext | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult: ...
    def position_qty(self) -> float: ...
    def equity(self) -> float: ...


@dataclass
class OrderResult:
    id: int
    status: str  # "filled" | "rejected"
    filled_qty: float
    avg_price: float
    error: Optional[str] = None

    def __getitem__(self, key: str):
        return getattr(self, key)


def submit_order(
    broker: OrderRouter,
    side: str,
    qty: float,
    price: Optional[float] = None,
    *,
    entry: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    ctx: TelemetryContext | None = None,
    client_order_id: str | None = None,
) -> OrderResult:
    """Submit a market order through ``broker`` logging the submission event."""

    if client_order_id is None:
        client_order_id = uuid.uuid4().hex

    if qty <= 0:
        log_event(
            E_ORDER_REJECTED,
            ctx,
            side=side,
            qty=qty,
            price=price,
            entry=entry,
            sl=sl,
            tp=tp,
            client_order_id=client_order_id,
            reason="invalid_qty",
        )
        return OrderResult(0, "rejected", 0.0, 0.0, "invalid_qty")

    if any(
        v is not None and not math.isfinite(v) for v in (entry, sl, tp)
    ):
        log_event(
            E_ORDER_REJECTED,
            ctx,
            side=side,
            qty=qty,
            price=price,
            entry=entry,
            sl=sl,
            tp=tp,
            client_order_id=client_order_id,
            reason="invalid_stops",
        )
        return OrderResult(0, "rejected", 0.0, 0.0, "invalid_stops")

    log_event(
        E_ORDER_SUBMITTED,
        ctx,
        side=side,
        qty=qty,
        price=price,
        entry=entry,
        sl=sl,
        tp=tp,
        client_order_id=client_order_id,
    )
    return broker.market_order(
        side,
        qty,
        price,
        entry=entry,
        sl=sl,
        tp=tp,
        ctx=ctx,
        client_order_id=client_order_id,
    )


class PaperBroker:
    """
    Bardzo prosty broker papierowy:
    - long/flat
    - prowizja procentowa od notional
    - equity = *zrealizowane* (cash), bez M2M — zgodnie z oczekiwaniem testów
    """

    def __init__(self, fee_perc: float = 0.0005) -> None:
        self._cash = 100_000.0
        self._pos = 0.0
        self._avg = 0.0
        self._last_price: Optional[float] = None
        self._connected = False
        self._id = 0
        self._fee_perc = float(fee_perc)

    # --- API ---------------------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def set_price(self, price: float) -> None:
        self._last_price = float(price)

    def position_qty(self) -> float:
        return self._pos

    def equity(self) -> float:
        # TYLKO zrealizowane – bez niezrealizowanego PnL
        return self._cash

    def _fee(self, notional: float) -> float:
        return abs(notional) * self._fee_perc

    def market_order(
        self,
        side: str,
        qty: float,
        price: Optional[float] = None,
        *,
        entry: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        ctx: TelemetryContext | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        self._id += 1

        log_event(
            E_ORDER_ACK,
            ctx,
            client_order_id=client_order_id,
            ticket=self._id,
        )

        if not self._connected:
            log_event(
                E_ORDER_REJECTED,
                ctx,
                client_order_id=client_order_id,
                ticket=self._id,
                reason="not connected",
            )
            return OrderResult(self._id, "rejected", 0.0, 0.0, "not connected")

        px = float(price) if price is not None else self._last_price
        if px is None:
            log_event(
                E_ORDER_REJECTED,
                ctx,
                client_order_id=client_order_id,
                ticket=self._id,
                reason="no price",
            )
            return OrderResult(self._id, "rejected", 0.0, 0.0, "no price")

        if qty <= 0:
            log_event(
                E_ORDER_REJECTED,
                ctx,
                client_order_id=client_order_id,
                ticket=self._id,
                reason="qty <= 0",
            )
            return OrderResult(self._id, "rejected", 0.0, 0.0, "qty <= 0")

        side_u = side.upper()

        if side_u == "BUY":
            notional = px * qty
            fee = self._fee(notional)
            # kupno: zmniejsza cash o koszt i fee, zwiększa pozycję
            self._cash -= notional + fee
            new_qty = self._pos + qty
            if new_qty > 0:
                self._avg = (self._avg * self._pos + notional) / new_qty
            self._pos = new_qty
            log_event(
                E_ORDER_FILLED,
                ctx,
                client_order_id=client_order_id,
                ticket=self._id,
                fill_price=px,
                fill_qty=qty,
            )
            return OrderResult(self._id, "filled", qty, px)

        if side_u == "SELL":
            # sprzedaż: pozwalamy sprzedać do wielkości pozycji (long/flat)
            sell_qty = min(qty, self._pos)
            if sell_qty <= 0:
                log_event(
                    E_ORDER_REJECTED,
                    ctx,
                    client_order_id=client_order_id,
                    ticket=self._id,
                    reason="no position to sell",
                )
                return OrderResult(self._id, "rejected", 0.0, 0.0, "no position to sell")
            notional = px * sell_qty
            fee = self._fee(notional)
            self._cash += notional - fee
            self._pos -= sell_qty
            if self._pos == 0.0:
                self._avg = 0.0
            log_event(
                E_ORDER_FILLED,
                ctx,
                client_order_id=client_order_id,
                ticket=self._id,
                fill_price=px,
                fill_qty=sell_qty,
            )
            return OrderResult(self._id, "filled", sell_qty, px)

        log_event(
            E_ORDER_REJECTED,
            ctx,
            client_order_id=client_order_id,
            ticket=self._id,
            reason=f"unknown side: {side}",
        )
        return OrderResult(self._id, "rejected", 0.0, 0.0, f"unknown side: {side}")
