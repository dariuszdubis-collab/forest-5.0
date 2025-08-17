import pandas as pd
from forest5.config import BacktestSettings, StrategySettings, RiskSettings
from forest5.backtest.engine import run_backtest

def test_engine_marks_equity_not_price():
    idx = pd.date_range("2024-01-01", periods=50, freq="h")
    close = pd.Series(1.10 + 0.001*pd.RangeIndex(len(idx)), index=idx)
    df = pd.DataFrame({
        "open": close.values,
        "high": close.values + 0.001,
        "low":  close.values - 0.001,
        "close": close.values,
    }, index=idx)
    df.index.name = "time"
    settings = BacktestSettings(
        symbol="EURUSD",
        strategy=StrategySettings(name="ema_cross", fast=5, slow=15, use_rsi=False),
        risk=RiskSettings(initial_capital=100_000.0, risk_per_trade=0.01,
                          fee_perc=0.0, slippage_perc=0.0),
        atr_period=14, atr_multiple=2.0,
    )
    res = run_backtest(df, settings)
    eq = res.equity_curve
    # jeżeli equity byłoby w skali ceny, średnia byłaby ~1.1-1.2; oczekujemy skali portfela
    assert eq.mean() > 10_000

