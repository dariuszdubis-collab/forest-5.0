from forest5.backtest.grid import run_grid
from forest5.examples.synthetic import generate_ohlc
import pandas as pd


def test_grid_small():
    df = generate_ohlc(periods=40, start_price=100.0, freq="D")
    res = run_grid(
        df,
        symbol="SYMB",
        fast_values=[6],
        slow_values=[12],
        capital=10_000.0,
        risk=0.01,
        n_jobs=1,
    )
    assert {"fast", "slow", "equity_end", "max_dd", "cagr", "rar"}.issubset(res.columns)
    assert len(res) == 1
    assert res["equity_end"].iloc[0] > 0


def test_grid_parallel_same_result():
    df = generate_ohlc(periods=40, start_price=100.0, freq="D")
    kwargs = dict(
        symbol="SYMB",
        fast_values=[6, 8],
        slow_values=[12, 20],
        capital=10_000.0,
        risk=0.01,
    )
    res1 = run_grid(df, n_jobs=1, **kwargs)
    res2 = run_grid(df, n_jobs=2, **kwargs)
    pd.testing.assert_frame_equal(
        res1.sort_values(["fast", "slow"]).reset_index(drop=True),
        res2.sort_values(["fast", "slow"]).reset_index(drop=True),
    )


def test_grid_multiple_parameters():
    df = generate_ohlc(periods=40, start_price=100.0, freq="D")
    res = run_grid(
        df,
        symbol="SYMB",
        fast_values=[6, 8],
        slow_values=[12, 20],
        risk_values=[0.01, 0.02],
        rsi_period_values=[14, 21],
        capital=10_000.0,
        n_jobs=1,
    )
    assert len(res) == 16  # 2*2*2*2 combinations
    assert set(res["risk"]) == {0.01, 0.02}
    assert set(res["rsi_period"]) == {14, 21}
