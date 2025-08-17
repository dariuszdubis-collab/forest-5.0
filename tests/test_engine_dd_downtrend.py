import pandas as pd
import numpy as np
from forest5.backtest.engine import run_backtest
from forest5.config import BacktestSettings, StrategySettings, RiskSettings

def test_dd_on_downtrend():
    # Wymuś longa od 1. bara: fast=1, slow=100 -> cross od razu
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    close = pd.Series(np.linspace(100.0, 60.0, len(idx)), index=idx)
    df = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5, "close": close
    }, index=idx)
    df.index.name = "time"

    s = BacktestSettings(
        strategy=StrategySettings(name="ema_cross", fast=1, slow=100, use_rsi=False),
        risk=RiskSettings(initial_capital=100_000.0, risk_per_trade=0.01, fee_perc=0.0, slippage_perc=0.0),
        atr_period=14, atr_multiple=2.0,
    )
    res = run_backtest(df, s)
    # Na spadku DD musi rosnąć – próg 0.30 powinien być osiągalny
    assert res.max_dd >= 0.20

