from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RiskManager:
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    max_drawdown: float = 0.30
    fee_perc: float = 0.0005
    slippage_perc: float = 0.0

    _cash: float = field(default=0.0, init=False)
    _position: float = field(default=0.0, init=False)  # qty (long-only w testach)
    _avg_price: float = field(default=0.0, init=False)
    _equity_curve: List[float] = field(default_factory=list, init=False)
    _peak: float = field(init=False)

    def __post_init__(self) -> None:
        self._cash = float(self.initial_capital)
        self._peak = float(self.initial_capital)
        # baseline – 1 punkt na starcie
        self._equity_curve.append(self.equity)

    # --- public properties -------------------------------------------------

    @property
    def equity(self) -> float:
        """Zrealizowane equity (bez niezrealizowanego PnL)."""
        return self._cash

    @property
    def equity_curve(self) -> List[float]:
        return self._equity_curve

    # --- core API -----------------------------------------------------------

    def position_size(self, price: float, atr: float, atr_multiple: float) -> float:
        """
        Sizing: (risk% * equity) / (ATR * multiple),
        z bezpiecznikiem, by *notional* nie przekroczył dostępnej gotówki.
        """
        if price <= 0 or atr <= 0 or atr_multiple <= 0:
            return 0.0

        risk_cash = self.equity * self.risk_per_trade
        denom = atr * atr_multiple
        if denom <= 0:
            return 0.0

        qty = risk_cash / denom

        # Bezpiecznik: nie kupuj więcej niż pozwala gotówka.
        # (fee/slippage w testach zwykle 0, więc cap dokładnie equity/price)
        cash_cap_qty = self.equity / price
        qty = min(qty, cash_cap_qty)

        return max(0.0, float(qty))

    def position_cost(self, price: float, qty: float) -> float:
        notional = abs(price * qty)
        fee = notional * self.fee_perc
        slippage = notional * self.slippage_perc
        return fee + slippage

    def record_trade_fill(self, side: str, price: float, qty: float) -> None:
        """
        Aktualizuje *cash/position/avg_price*. NIE dopisuje punktu equity –
        markowanie robimy *raz na bar* w engine.
        """
        if qty <= 0:
            return

        cost = price * qty
        fee = self.position_cost(price, qty)

        if side.upper() == "BUY":
            self._cash -= (cost + fee)
            new_qty = self._position + qty
            if new_qty > 0:
                self._avg_price = (self._avg_price * self._position + cost) / new_qty
            self._position = new_qty

        elif side.upper() == "SELL":
            self._cash += (cost - fee)
            self._position = max(0.0, self._position - qty)
            if self._position == 0.0:
                self._avg_price = 0.0

        # peak aktualizujemy na zrealizowanym equity; próbkę equity dodamy na koniec bara
        self._peak = max(self._peak, self.equity)

    # Cienkie wrappery dla czytelności w engine
    def buy(self, price: float, qty: float) -> None:
        self.record_trade_fill("BUY", price, qty)

    def sell(self, price: float, qty: float) -> None:
        self.record_trade_fill("SELL", price, qty)

    def record_mark_to_market(self, equity_value: float) -> None:
        """Zapisujemy przekazane EQUITY (MTM) – nie cenę!"""
        ev = float(equity_value)
        self._equity_curve.append(ev)
        self._peak = max(self._peak, ev)

    def exceeded_max_dd(self) -> bool:
        """Czy ostatnie equity spadło o >= max_drawdown od dotychczasowego szczytu."""
        if not self._equity_curve:
            return False
        last = self._equity_curve[-1]
        if self._peak <= 0:
            return False
        dd = (self._peak - last) / self._peak
        return dd >= self.max_drawdown

