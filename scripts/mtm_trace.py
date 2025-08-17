import numpy as np
import pandas as pd
from forest5.config import BacktestSettings, StrategySettings, RiskSettings
from forest5.backtest.engine import run_backtest

idx = pd.date_range("2024-01-01", periods=50, freq="h")
close = pd.Series(1.12 + 0.0001*np.arange(len(idx)), index=idx)
df = pd.DataFrame({"open": close, "high": close+0.0002, "low": close-0.0002, "close": close})
df.index.name = "time"

s = BacktestSettings(
    strategy=StrategySettings(name="ema_cross", fast=12, slow=26, use_rsi=False),
    risk=RiskSettings(initial_capital=100_000.0, risk_per_trade=0.01, fee_perc=0.0, slippage_perc=0.0),
    atr_period=14, atr_multiple=2.0,
)

res = run_backtest(df, s)
print(f"bars: {len(df)} equity marks: {len(res.equity_curve)}")
print(res.equity_curve.head(12))

