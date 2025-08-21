import numpy as np
import pandas as pd
from forest5.config import BacktestSettings, StrategySettings, RiskSettings
from forest5.backtest.engine import run_backtest

idx = pd.date_range("2024-01-01", periods=50, freq="h")
close = pd.Series(1.12 + 0.0001 * np.arange(len(idx)), index=idx)
df = pd.DataFrame({"open": close, "high": close + 0.0002, "low": close - 0.0002, "close": close})
df.index.name = "time"

s = BacktestSettings(
    strategy=StrategySettings(name="ema_cross", fast=12, slow=26, use_rsi=False),
    risk=RiskSettings(
        initial_capital=100_000.0, risk_per_trade=0.01, fee_perc=0.0, slippage_perc=0.0
    ),
    atr_period=14,
    atr_multiple=2.0,
)

res = run_backtest(df, s)

equity = res.equity_curve
equity_end = float(equity.iloc[-1]) if not equity.empty else 0.0
ret = equity_end / float(s.risk.initial_capital) - 1.0
print(f"bars: {len(df)} equity marks: {len(equity)}")
print(equity.head(12))
print(f"return: {ret:.6f} max_dd: {res.max_dd:.6f} trades: {len(res.trades.trades)}")
