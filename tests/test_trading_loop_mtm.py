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


def test_trading_loop_marks_once_per_bar():
    idx = pd.date_range("2024-01-01", periods=20, freq="h")
    close = pd.Series(1.12 + 0.0001 * np.arange(len(idx)), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close + 0.0002, "low": close - 0.0002, "close": close},
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

    tb = TradeBook()
    rm = RiskManager(**settings.risk.model_dump())
    pos = bootstrap_position(df, sig, rm, tb, settings, "close", settings.atr_multiple)
    _trading_loop(df, sig, rm, tb, pos, "close", settings.atr_multiple, settings)

    assert len(rm.equity_curve) == len(df) + 1
