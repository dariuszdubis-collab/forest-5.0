import pandas as pd

from forest5.backtest.grid import run_param_grid
from forest5.config import BacktestSettings
from forest5.examples.synthetic import generate_ohlc


def _base_settings() -> BacktestSettings:
    return BacktestSettings(symbol="SYMB", timeframe="1h", strategy={"name": "ema_cross"})


def test_grid_export_columns(tmp_path):
    df = generate_ohlc(periods=10, start_price=100.0, freq="D")
    settings = _base_settings()
    params = {"fast": [5], "slow": [10]}
    results_path = tmp_path / "results.csv"
    meta_path = tmp_path / "meta.json"
    run_param_grid(
        df,
        settings,
        params,
        results_path=results_path,
        meta_path=meta_path,
    )
    assert results_path.exists()
    csv = pd.read_csv(results_path)
    expected_cols = [
        "fast",
        "slow",
        "equity_end",
        "dd",
        "cagr",
        "rar",
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
    ]
    assert list(csv.columns) == expected_cols
