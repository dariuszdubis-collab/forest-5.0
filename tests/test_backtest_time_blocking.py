import pandas as pd

from forest5.backtest.engine import run_backtest
from forest5.config import BacktestSettings


def test_time_blocked_logged(capfd):
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
        },
        index=idx,
    )
    settings = BacktestSettings()
    settings.time.blocked_hours = [1]
    run_backtest(df, settings)
    out = capfd.readouterr().out
    assert "time_blocked" in out
