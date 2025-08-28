import warnings
import pandas as pd
import numpy as np

from forest5.config import BacktestSettings
from forest5.grid.engine import plan_param_grid, run_grid


def test_grid_no_settingwithcopy():
    warnings.simplefilter("error", pd.errors.SettingWithCopyWarning)
    idx = pd.date_range("2021-01-01", periods=10, freq="h")
    prices = np.linspace(1, 10, 10)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
        },
        index=idx,
    )
    df = df.loc[df.index > idx[0]]  # slice without copy

    base = BacktestSettings(symbol="EURUSD", timeframe="1h")
    combos = plan_param_grid(
        {"ema_fast": [5, 6], "ema_slow": [10, 12]},
        filter_fn=lambda c: c["ema_fast"] < c["ema_slow"],
    )
    res = run_grid(df, combos.head(2), base)
    assert not res.empty
