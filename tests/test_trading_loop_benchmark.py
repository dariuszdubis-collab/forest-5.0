import timeit

import numpy as np
import pandas as pd

from forest5.backtest.engine import (
    _generate_signal,
    _trading_loop,
    _validate_data,
    bootstrap_position,
)
from forest5.backtest.risk import RiskManager
from forest5.backtest.tradebook import TradeBook
from forest5.config import BacktestSettings, RiskSettings, StrategySettings
from forest5.core.indicators import atr


def _old_trading_loop(df, sig, rm, tb, position, price_col, atr_multiple):
    for t, row in df.iterrows():
        price = float(row[price_col])
        this_sig = int(sig.loc[t]) if t in sig.index else 0

        if this_sig < 0 and position > 0.0:
            rm.sell(price, position)
            tb.add(t, price, position, "SELL")
            position = 0.0

        if this_sig > 0 and position <= 0.0:
            qty = rm.position_size(
                price=price, atr=float(row["atr"]), atr_multiple=atr_multiple
            )
            if qty > 0.0:
                rm.buy(price, qty)
                tb.add(t, price, qty, "BUY")
                position = qty

        equity_mtm = rm.equity + position * price
        rm.record_mark_to_market(equity_mtm)

        if rm.exceeded_max_dd():
            break
    return position


def _setup():
    n = 5000
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 1.12 + 0.0001 * np.arange(n)
    df = pd.DataFrame(
        {"open": base, "high": base + 0.0002, "low": base - 0.0002, "close": base},
        index=idx,
    )
    df.index.name = "time"

    settings = BacktestSettings(
        strategy=StrategySettings(name="ema_cross", fast=12, slow=26, use_rsi=False),
        risk=RiskSettings(
            initial_capital=100_000.0,
            risk_per_trade=0.01,
            fee_perc=0.0,
            slippage_perc=0.0,
        ),
        atr_period=1,
        atr_multiple=2.0,
    )

    df = _validate_data(df, price_col="close")
    sig = _generate_signal(df, settings, price_col="close")
    df["atr"] = atr(df["high"], df["low"], df["close"], settings.atr_period)
    return df, sig, settings


def test_trading_loop_benchmark():
    df, sig, settings = _setup()

    def run_old():
        tb = TradeBook()
        rm = RiskManager(**settings.risk.model_dump())
        pos = bootstrap_position(df, sig, rm, tb, settings, "close", settings.atr_multiple)
        _old_trading_loop(df, sig, rm, tb, pos, "close", settings.atr_multiple)

    def run_new():
        tb = TradeBook()
        rm = RiskManager(**settings.risk.model_dump())
        pos = bootstrap_position(df, sig, rm, tb, settings, "close", settings.atr_multiple)
        _trading_loop(df, sig, rm, tb, pos, "close", settings.atr_multiple)

    t_old = timeit.timeit(run_old, number=3)
    t_new = timeit.timeit(run_new, number=3)
    assert t_new <= t_old
