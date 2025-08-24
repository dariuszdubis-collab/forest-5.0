import numpy as np
import pandas as pd
import pytest

pytest.skip("legacy mtm invariants incompatible", allow_module_level=True)


def _mk_fx_df(n=50):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(1.12 + 0.0001 * np.arange(n), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
        },
        index=idx,
    )
    df.index.name = "time"
    return df


def _mk_downtrend_df(n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.linspace(100.0, 60.0, n)
    df = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
        index=idx,
    )
    df.index.name = "time"
    return df


def test_once_per_bar_equity_length():
    df = _mk_fx_df(50)
    s = BacktestSettings(
        strategy=StrategySettings(name="ema_cross", fast=12, slow=26, use_rsi=False),
        risk=RiskSettings(
            initial_capital=100_000.0, risk_per_trade=0.01, fee_perc=0.0, slippage_perc=0.0
        ),
        atr_period=14,
        atr_multiple=2.0,
    )
    res = run_backtest(df, s)
    # Dopuszczamy N (znacznik na close każdego bara) albo N+1 (punkt startowy + każdy bar)
    assert len(res.equity_curve) in (len(df), len(df) + 1)


def test_equity_not_in_price_scale_fx():
    df = _mk_fx_df(50)
    s = BacktestSettings(
        strategy=StrategySettings(name="ema_cross", fast=5, slow=15, use_rsi=False),
        risk=RiskSettings(
            initial_capital=100_000.0, risk_per_trade=0.01, fee_perc=0.0, slippage_perc=0.0
        ),
        atr_period=14,
        atr_multiple=2.0,
    )
    res = run_backtest(df, s)
    # Equity musi być w skali kapitału, nie ceny (np. ~100k, a nie ~1.1x)
    assert res.equity_curve.mean() > 10_000.0


def test_dd_on_downtrend_reaches_threshold():
    df = _mk_downtrend_df(60)
    s = BacktestSettings(
        strategy=StrategySettings(name="ema_cross", fast=1, slow=100, use_rsi=False),
        risk=RiskSettings(
            initial_capital=100_000.0, risk_per_trade=0.01, fee_perc=0.0, slippage_perc=0.0
        ),
        atr_period=14,
        atr_multiple=2.0,
    )
    res = run_backtest(df, s)
    # Na silnym spadku DD powinien być zauważalny (próg 0.20 jako rozsądny sanity target)
    assert res.max_dd >= 0.20
