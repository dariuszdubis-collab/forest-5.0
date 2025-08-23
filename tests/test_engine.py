import numpy as np
import pandas as pd
from unittest.mock import patch

from forest5.backtest.engine import run_backtest
from forest5.config import BacktestSettings


def _capture_atr(storage):
    def _inner(high, low, close, period):
        from forest5.core.indicators import atr as real_atr

        result = real_atr(high, low, close, period)
        storage[period] = result
        return result

    return _inner


def test_atr_period_override_changes_atr():
    idx = pd.date_range("2020", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [9.6, 10.6, 11.6, 12.6, 13.6],
            "high": [10, 11, 12, 13, 14],
            "low": [9, 10, 11, 12, 13],
            "close": [9.5, 10.5, 11.5, 12.5, 13.5],
        },
        index=idx,
    )
    settings = BacktestSettings()

    storage: dict[int, np.ndarray] = {}
    with patch("forest5.backtest.engine.atr", side_effect=_capture_atr(storage)):
        run_backtest(df, settings)
    default_atr = storage[settings.atr_period]

    storage = {}
    with patch("forest5.backtest.engine.atr", side_effect=_capture_atr(storage)):
        run_backtest(df, settings, atr_period=1)
    override_atr = storage[1]

    assert not np.array_equal(override_atr, default_atr)
