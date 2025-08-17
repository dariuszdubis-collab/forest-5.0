from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class Trade:
    time: Any
    price: float
    qty: float
    side: str  # "BUY" | "SELL"


class TradeBook:
    def __init__(self) -> None:
        self.trades: List[Trade] = []

    def add(self, time, price: float, qty: float, side: str):
        self.trades.append(Trade(time=time, price=float(price), qty=float(qty), side=side))

    def __len__(self) -> int:
        return len(self.trades)
