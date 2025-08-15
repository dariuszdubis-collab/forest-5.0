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
    _position: float = field(default=0.0, init=False)  # qty
    _avg_price: float = field(default=0.0, init=False)
    _equity_curve: List[float] = field(default_factory=list, init=False)
    _peak: float = field(init=False)

    def __post_init__(self) -> None:
        self._cash = float(self.initial_capital)
        self._peak = float(self.initial_capital)
        self._equity_curve.append(self.equity)

    # --- public properties -------------------------------------------------

    @property
    def equity(self) -> float:
        """Aktualne equity wg *zrealizowanych* wyników (bez M2M)."""
        return self._cash

    @property
    def equity_curve(self) -> List[float]:
        return self._equity_curve

    # --- core API -----------------------------------------------------------

    def position_size(self, price: float, atr: float, atr_multiple: float) -> float:
        """
        Sizing: (risk% * equity) / (ATR * multiple)
        """
        if atr <= 0 or atr_multiple <= 0 or price <= 0:
            return 0.0
        risk_cash = self.equity * self.risk_per_trade
        denom = atr * atr_multiple
        if denom <= 0:
            return 0.0
        qty = risk_cash / denom
        return max(0.0, qty)

    def position_cost(self, price: float, qty: float) -> float:
        notional = abs(price * qty)
        spread = 0.0  # opcjonalnie do dodania później
        fee = notional * self.fee_perc
        slippage = notional * self.slippage_perc
        return spread + fee + slippage

    # Zdarzenia realizujące wynik (BUY/SELL) aktualizują _cash (zysk/strata zrealizowane)
    # i *nie* dopisują niezrealizowanego PnL do equity.

    def record_trade_fill(self, side: str, price: float, qty: float) -> None:
        """
        Symulacja wpływu transakcji na _cash oraz średnią cenę pozycji.
        Zakładamy long-only (qty >= 0) na potrzeby testów.
        """
        if qty <= 0:
            return
        cost = price * qty
        fee = self.position_cost(price, qty)

        if side.upper() == "BUY":
            # zmniejszamy cash o koszt i prowizję
            self._cash -= (cost + fee)
            # aktualizujemy średnią cenę pozycji
            new_qty = self._position + qty
            if new_qty > 0:
                self._avg_price = (self._avg_price * self._position + cost) / new_qty
            self._position = new_qty
        elif side.upper() == "SELL":
            # zwiększamy cash o wpływy i odejmujemy fee
            self._cash += (cost - fee)
            self._position = max(0.0, self._position - qty)
            if self._position == 0.0:
                self._avg_price = 0.0

        # equity (zrealizowane) po transakcji
        self._equity_curve.append(self.equity)
        self._peak = max(self._peak, self.equity)

    def record_mark_to_market(self, equity_value: float) -> None:
        """
        Zapisz *podane equity* do krzywej (to jest zgodne z testem),
        zaktualizuj peak i pozwól exceeded_max_dd() wykryć DD.
        """
        self._equity_curve.append(float(equity_value))
        self._peak = max(self._peak, float(equity_value))

    def exceeded_max_dd(self) -> bool:
        """
        Czy ostatnie equity spadło o >= max_drawdown od dotychczasowego szczytu.
        """
        if not self._equity_curve:
            return False
        last = self._equity_curve[-1]
        if self._peak <= 0:
            return False
        dd = (self._peak - last) / self._peak
        return dd >= self.max_drawdown

