import numpy as np
import pandas as pd
import pytest

from forest5.backtest.engine import run_backtest
from forest5.config import BacktestSettings
from forest5.time_only import TimeOnlyModel


@pytest.mark.timeonly
def test_backtest_timeonly_ab(tmp_path) -> None:
    # Generate 250 minute bars with slight trending variation
    idx = pd.date_range("2024-01-01", periods=250, freq="1min")
    close_up = np.linspace(100, 101, 125)
    close_down = np.linspace(101, 99, 125)
    close = np.concatenate([close_up, close_down])
    close += np.sin(np.linspace(0, 20, 250)) * 0.05
    open_ = close - 0.05
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    # Baseline
    settings_base = BacktestSettings()
    settings_base.time.model.enabled = False
    settings_base.time.fusion_min_confluence = 1
    res_base = run_backtest(df, settings_base)
    trades_base = len(res_base.trades)

    # TimeOnly variant
    gates = {h: (30.0, 90.0) for h in range(24)}
    model = TimeOnlyModel(gates, q_low=0.0, q_high=1.0)
    model_path = tmp_path / "time_only.json"
    model.save(model_path)
    settings_time = BacktestSettings()
    settings_time.time.model.enabled = True
    settings_time.time.model.path = model_path
    settings_time.time.fusion_min_confluence = 2
    res_time = run_backtest(df, settings_time)
    trades_time = len(res_time.trades)

    assert trades_time <= trades_base
