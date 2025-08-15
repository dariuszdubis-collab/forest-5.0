from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


class OrderRouter(Protocol):
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def set_price(self, price: float) -> None: ...
    def market_order(self, side: str, qty: float, price: Optional[float] = None) -> OrderResult: ...
    def position_qty(self) -> float: ...
    def equity(self) -> float: ...


@dataclass
class OrderResult:
    id: int
    status: str  # "filled" | "rejected"
    filled_qty: float
    avg_price: float
    error: Optional[str] = None


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

    def market_order(self, side: str, qty: float, price: Optional[float] = None) -> OrderResult:
        self._id += 1

        if not self._connected:
            return OrderResult(self._id, "rejected", 0.0, 0.0, "not connected")

        px = float(price) if price is not None else self._last_price
        if px is None:
            return OrderResult(self._id, "rejected", 0.0, 0.0, "no price")

        if qty <= 0:
            return OrderResult(self._id, "rejected", 0.0, 0.0, "qty <= 0")

        side_u = side.upper()
        notional = px * qty
        fee = self._fee(notional)

        if side_u == "BUY":
            # kupno: zmniejsza cash o koszt i fee, zwiększa pozycję
            self._cash -= (notional + fee)
            new_qty = self._pos + qty
            if new_qty > 0:
                self._avg = (self._avg * self._pos + notional) / new_qty
            self._pos = new_qty
            return OrderResult(self._id, "filled", qty, px)

        if side_u == "SELL":
            # sprzedaż: pozwalamy sprzedać do wielkości pozycji (long/flat)
            sell_qty = min(qty, self._pos)
            if sell_qty <= 0:
                return OrderResult(self._id, "rejected", 0.0, 0.0, "no position to sell")
            self._cash += (px * sell_qty - fee)
            self._pos -= sell_qty
            if self._pos == 0.0:
                self._avg = 0.0
            return OrderResult(self._id, "filled", sell_qty, px)

        return OrderResult(self._id, "rejected", 0.0, 0.0, f"unknown side: {side}")

