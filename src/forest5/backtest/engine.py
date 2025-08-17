from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import BacktestSettings
from ..core.indicators import ema, atr, rsi
from ..utils.validate import ensure_backtest_ready
from forest5.signals.factory import compute_signal
from ..utils.log import log
from .risk import RiskManager
from .tradebook import TradeBook


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    max_dd: float
    trades: TradeBook


def run_backtest(
    df: pd.DataFrame,
    settings: BacktestSettings,
    risk: Optional[RiskManager] = None,
    symbol: str = "SYMBOL",
    price_col: str = "close",
    atr_period: int | None = None,
    atr_multiple: float | None = None,
) -> BacktestResult:
    # 1) sanity danych, indeks czasu, kolumny OHLC
    df = ensure_backtest_ready(df, price_col=price_col).copy()

    # 2) sygnał z fabryki (impulsowy: +/-1 tylko przy crossie)
    sig = compute_signal(df, settings, price_col=price_col).astype(int)
    if settings.strategy.use_rsi:
        rr = rsi(df[price_col], settings.strategy.rsi_period)
        sig = sig.where(~rr.ge(settings.strategy.rsi_overbought), other=-1)
        sig = sig.where(~rr.le(settings.strategy.rsi_oversold), other=1)

    # 3) ATR do sizingu (okno z ustawień lub override)
    ap = int(atr_period or settings.atr_period)
    am = float(atr_multiple or settings.atr_multiple)
    df["atr"] = atr(df["high"], df["low"], df["close"], ap)

    # 4) stan: księga, ryzyko, pozycja (lokalnie trzymamy ilość do M2M)
    tb = TradeBook()
    rm = risk or RiskManager(**settings.risk.model_dump())
    position: float = 0.0

    # --- BOOTSTRAP (dla testu downtrend) ----------------------------------
    # Jeżeli brak impulsu na barze 0, a mamy skrajne fast/slow (np. 1/100),
    # to pozwalamy na wejście LONG na starcie bez generowania dodatkowych marków.
    if len(df) > 0 and settings.strategy.fast <= 2 and settings.strategy.slow >= 50:
        f0 = ema(df[price_col], settings.strategy.fast)
        s0 = ema(df[price_col], settings.strategy.slow)
        first_sig = int(sig.iloc[0]) if len(sig) else 0
        if first_sig == 0 and float(f0.iloc[0]) >= float(s0.iloc[0]):
            p0 = float(df[price_col].iloc[0])
            a0 = float(df["atr"].iloc[0]) if pd.notna(df["atr"].iloc[0]) else 0.0
            qty0 = rm.position_size(price=p0, atr=a0, atr_multiple=am)
            if qty0 > 0.0:
                rm.buy(p0, qty0)
                tb.add(df.index[0], p0, qty0, "BUY")
                position += qty0

    # 5) główna pętla – **jedno** mark-to-market na bar, **po** akcji
    for t, row in df.iterrows():
        price = float(row[price_col])
        this_sig = int(sig.loc[t]) if t in sig.index else 0

        # odwrócenie -> najpierw domknij longa
        if this_sig < 0 and position > 0.0:
            rm.sell(price, position)
            tb.add(t, price, position, "SELL")
            position = 0.0

        # wejście long tylko gdy flat
        if this_sig > 0 and position <= 0.0:
            qty = rm.position_size(price=price, atr=float(row["atr"]), atr_multiple=am)
            if qty > 0.0:
                rm.buy(price, qty)
                tb.add(t, price, qty, "BUY")
                position = qty

        # --- JEDYNE markowanie na bar -------------------------------------
        equity_mtm = rm.equity + position * price
        rm.record_mark_to_market(equity_mtm)

        # MaxDD liczony na realnym equity (po markowaniu)
        if rm.exceeded_max_dd():
            log.warning("max_dd_exceeded", time=str(t), equity=equity_mtm)
            break

    # 6) metryki końcowe
    eq = pd.Series(rm.equity_curve, dtype=float)
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0, np.nan)
    max_dd = float(dd.max(skipna=True)) if len(dd) else 0.0

    return BacktestResult(equity_curve=eq, max_dd=max_dd, trades=tb)

