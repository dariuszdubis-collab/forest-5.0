import pandas as pd
from forest5.examples.synthetic import generate_ohlc
from forest5.backtest.grid import run_grid
import pytest


def test_run_grid_hourly_regression():
    """Run a small grid search on hourly data and check deterministic output."""
    df = generate_ohlc(periods=40, start_price=100.0, freq="h")
    res = run_grid(
        df,
        symbol="SYMB",
        fast_values=[6],
        slow_values=[12],
        capital=10_000.0,
        risk=0.01,
        n_jobs=1,
    )
    assert len(res) == 1
    # regression check for equity end ensures correct timeframe (1h)
    assert res["equity_end"].iloc[0] == pytest.approx(9989.292229, rel=1e-6)
