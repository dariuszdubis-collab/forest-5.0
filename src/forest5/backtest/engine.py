from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import BacktestSettings
from ..core.indicators import ema, atr, rsi
from ..utils.validate import ensure_backtest_ready
from ..utils.log import log
from .risk import RiskManager
from .tradebook import TradeBook


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    max_dd: float
    trades: TradeBook


def ema_cross_signal(close: pd.Series, fast: int, slow: int) -> pd.Series:
    f = ema(close, fast)
    s = ema(close, slow)
    raw = np.where(f > s, 1, -1)
    sig = pd.Series(raw, index=close.index, dtype=int)
    # sygnał tylko na zmiany kierunku (cross)
    cross = sig.ne(sig.shift(1)).astype(int)
    # 1/-1 przy zmianie, w innym wypadku 0
    return sig.where(cross.eq(1), other=0)


def run_backtest(
    df: pd.DataFrame,
    settings: BacktestSettings,
    risk: Optional[RiskManager] = None,
    symbol: str = "SYMBOL",
    price_col: str = "close",
    atr_period: int | None = None,
    atr_multiple: float | None = None,
) -> BacktestResult:
    df = ensure_backtest_ready(df, price_col=price_col).copy()

    # sygnały
    sig = ema_cross_signal(df[price_col], settings.strategy.fast, settings.strategy.slow)
    if settings.strategy.use_rsi:
        r = rsi(df[price_col], settings.strategy.rsi_period)
        sig = sig.where(~r.ge(settings.strategy.rsi_overbought), other=-1)
        sig = sig.where(~r.le(settings.strategy.rsi_oversold), other=1)

    # ATR
    ap = atr_period or settings.atr_period
    am = atr_multiple or settings.atr_multiple
    df["atr"] = atr(df["high"], df["low"], df["close"], ap)

    tb = TradeBook()
    rm = risk or RiskManager(**settings.risk.model_dump())
    position = 0.0

    for t, row in df.iterrows():
        price = float(row[price_col])
        this_sig = int(sig.loc[t]) if t in sig.index else 0

        # trail / maxDD check: mark-to-market
        rm.record_mark_to_market(price)
        if rm.exceeded_max_dd():
            log.warning("max_dd_exceeded", time=str(t), equity=rm.equity)
            break

        if this_sig != 0:
            # zamknij jeśli odwrotny sygnał
            if position > 0 and this_sig < 0:
                rm.sell(price, position)
                tb.add(t, price, position, "SELL")
                position = 0.0

            # otwórz long jeśli BUY
            if this_sig > 0 and position == 0.0:
                qty = rm.position_size(price=price, atr=float(row["atr"]), atr_multiple=am)
                if qty > 0:
                    rm.buy(price, qty)
                    tb.add(t, price, qty, "BUY")
                    position = qty

        # mark-to-market po akcji
        rm.record_mark_to_market(price)

    eq = pd.Series(rm.equity_curve, index=range(len(rm.equity_curve)), dtype=float)
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0, np.nan)
    max_dd = float(dd.max(skipna=True)) if len(dd) else 0.0
    return BacktestResult(equity_curve=eq, max_dd=max_dd, trades=tb)

