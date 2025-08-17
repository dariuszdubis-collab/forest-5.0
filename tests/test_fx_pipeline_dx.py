from __future__ import annotations
import pandas as pd
import numpy as np

from forest5.utils.validate import ensure_backtest_ready
from forest5.config import BacktestSettings, StrategySettings, RiskSettings
from forest5.backtest.engine import run_backtest


def _mk_df_with_time(colname: str) -> pd.DataFrame:
    # mini ramka OHLC + kolumna czasu pod różnymi nazwami
    idx = pd.date_range("2020-01-01", periods=6, freq="h")
    df = pd.DataFrame(
        {
            colname: idx,  # time/date/datetime/timestamp
            "open": [1.1000, 1.1010, 1.1020, 1.1030, 1.1040, 1.1050],
            "high": [1.1015, 1.1025, 1.1035, 1.1045, 1.1055, 1.1060],
            "low": [1.0990, 1.1005, 1.1010, 1.1020, 1.1030, 1.1040],
            "close": [1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1055],
        }
    )
    return df


def test_csv_time_detection_variants():
    for c in ("time", "date", "datetime", "timestamp"):
        df = _mk_df_with_time(c)
        df = ensure_backtest_ready(df, price_col="close")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing
        assert not df.index.has_duplicates


def test_equity_not_in_price_scale(monkeypatch):
    # syntetyk w skali FX ~1.1-1.2, equity powinno być ~ initial_capital
    idx = pd.date_range("2020-01-01", periods=50, freq="h")
    close = pd.Series(np.linspace(1.10, 1.20, len(idx)), index=idx)
    df = pd.DataFrame(
        {
            "open": close.values,
            "high": close.values + 0.001,
            "low": close.values - 0.001,
            "close": close.values,
        },
        index=idx,
    )
    df.index.name = "time"
    df = ensure_backtest_ready(df)

    # proste ustawienia: sygnał ema_cross, brak RSI
    settings = BacktestSettings(
        symbol="EURUSD",
        strategy=StrategySettings(name="ema_cross", fast=5, slow=15, use_rsi=False),
        risk=RiskSettings(
            initial_capital=100_000.0, risk_per_trade=0.01, fee_perc=0.0, slippage_perc=0.0
        ),
        atr_period=14,
        atr_multiple=2.0,
    )

    res = run_backtest(df, settings)
    eq = res.equity_curve
    assert eq.mean() > 10_000, "Krzywa equity wygląda na skalę ceny, a nie portfela."


def test_fx_position_size_reasonable():
    # ATR rzędu 0.005-0.01 na H1 → wielkości rzędu dziesiątek- setek tysięcy units są normalne
    from forest5.backtest.risk import RiskManager

    rm = RiskManager(initial_capital=100_000.0, risk_per_trade=0.01)
    qty = rm.position_size(price=1.10, atr=0.005, atr_multiple=2.0)
    # 1% z 100k = 1000; denom=0.005*2=0.01 → qty ~ 100000
    assert 50_000 <= qty <= 200_000, f"Unexpected qty={qty}"
