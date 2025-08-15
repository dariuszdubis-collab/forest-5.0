from forest5.backtest.grid import run_grid
from forest5.examples.synthetic import generate_ohlc


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

