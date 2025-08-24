import pandas as pd

from forest5.backtest.grid import run_param_grid
from forest5.config import BacktestSettings
from forest5.examples.synthetic import generate_ohlc


def _base_settings() -> BacktestSettings:
    return BacktestSettings(symbol="SYMB", timeframe="1h", strategy={"name": "ema_cross"})


def test_param_grid_dry_run(tmp_path):
    df = generate_ohlc(periods=10, start_price=100.0, freq="D")
    settings = _base_settings()
    params = {"fast": [5], "slow": [10]}
    res, meta = run_param_grid(df, settings, params, dry_run=True)
    assert len(res) == 1
    assert res.iloc[0]["fast"] == 5
    assert meta["n_combos"] == 1


def test_param_grid_parallel_same_result():
    df = generate_ohlc(periods=40, start_price=100.0, freq="D")
    settings = _base_settings()
    params = {"fast": [6, 8], "slow": [12, 20]}
    res1, _ = run_param_grid(df, settings, params, jobs=1, seed=123)
    res2, _ = run_param_grid(df, settings, params, jobs=2, seed=123)
    pd.testing.assert_frame_equal(
        res1.sort_values(["fast", "slow"]).reset_index(drop=True),
        res2.sort_values(["fast", "slow"]).reset_index(drop=True),
    )


def test_param_grid_metrics_columns():
    df = generate_ohlc(periods=40, start_price=100.0, freq="D")
    settings = _base_settings()
    params = {"fast": [6], "slow": [12]}
    res, meta = run_param_grid(df, settings, params)
    expected = {
        "trades",
        "winrate",
        "pnl",
        "pnl_net",
        "sharpe",
        "expectancy",
        "expectancy_by_pattern",
        "timeonly_wait_pct",
        "setups_expired_pct",
        "rr_avg",
        "rr_median",
    }
    assert expected.issubset(res.columns)
    assert meta["n_combos"] == 1
