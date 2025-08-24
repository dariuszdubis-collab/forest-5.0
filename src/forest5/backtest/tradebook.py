from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class Trade:
    time: Any
    price_open: float
    price_close: float
    qty: float
    side: str  # "BUY" | "SELL"
    pnl: float
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    reason_close: Optional[str] = None
    setup_id: Optional[str] = None
    pattern: Optional[str] = None


class TradeBook:
    columns = [
        "time",
        "price_open",
        "price_close",
        "qty",
        "side",
        "pnl",
        "entry",
        "sl",
        "tp",
        "reason_close",
        "setup_id",
        "pattern",
    ]

    def __init__(self) -> None:
        self.trades: List[Trade] = []

    def add(
        self,
        time,
        price_open: float,
        qty: float,
        side: str,
        price_close: float | None = None,
        entry: float | None = None,
        sl: float | None = None,
        tp: float | None = None,
        reason_close: str | None = None,
        setup_id: str | None = None,
        pattern: str | None = None,
    ) -> None:
        price_open = float(price_open)
        price_close = float(price_close if price_close is not None else price_open)
        qty = float(qty)
        side_u = side.upper()
        pnl = (price_close - price_open) * qty if side_u == "BUY" else (price_open - price_close) * qty
        self.trades.append(
            Trade(
                time=time,
                price_open=price_open,
                price_close=price_close,
                qty=qty,
                side=side_u,
                pnl=pnl,
                entry=entry,
                sl=sl,
                tp=tp,
                reason_close=reason_close,
                setup_id=setup_id,
                pattern=pattern,
            )
        )

    def __len__(self) -> int:
        return len(self.trades)

    def to_frame(self):  # pragma: no cover - simple export helper
        try:
            import pandas as pd
        except Exception:  # pragma: no cover - defensive
            raise RuntimeError("pandas required for exporting trades")
        return pd.DataFrame([t.__dict__ for t in self.trades], columns=self.columns)
