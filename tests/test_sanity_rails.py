import numpy as np
import pandas as pd
from forest5.backtest.engine import run_backtest
from forest5.config import BacktestSettings, StrategySettings, RiskSettings


def _mk_df_trend(n=60, start=100.0, end=60.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    c = np.linspace(start, end, n)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c}, index=idx)


def test_once_per_bar_and_dd():
    df = _mk_df_trend()
    s = BacktestSettings(
        strategy=StrategySettings(name="ema_cross", fast=1, slow=100, use_rsi=False),
        risk=RiskSettings(
            initial_capital=100_000.0, risk_per_trade=0.01, fee_perc=0.0, slippage_perc=0.0
        ),
        atr_period=14,
        atr_multiple=2.0,
    )
    res = run_backtest(df, s)
    # długość: N (+1 jeśli startowa kropka)
    assert len(res.equity_curve) in (len(df), len(df) + 1)
    # DD realny:
    peak = res.equity_curve.cummax()
    dd = (peak - res.equity_curve) / peak.replace(0, np.nan)
    assert dd.max() >= 0.20
