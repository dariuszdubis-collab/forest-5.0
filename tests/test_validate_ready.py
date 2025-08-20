# tests/test_validate_ready.py
import pandas as pd
from forest5.utils.validate import ensure_backtest_ready


def test_validate_accepts_time_column_and_makes_index():
    df = pd.DataFrame(
        {
            "time": ["2020-01-01 00:00", "2020-01-01 01:00"],
            "open": [1.1, 1.2],
            "high": [1.2, 1.3],
            "low": [1.0, 1.1],
            "close": [1.15, 1.25],
        }
    )
    out = ensure_backtest_ready(df, price_col="close")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing
    assert set(["open", "high", "low", "close"]).issubset(out.columns)


def test_validate_does_not_modify_input_df():
    df = pd.DataFrame(
        {
            "time": ["2020-01-01 00:00", "2020-01-01 01:00"],
            "open": [1.1, 1.2],
            "high": [1.2, 1.3],
            "low": [1.0, 1.1],
            "close": [1.15, 1.25],
        }
    )
    original = df.copy()
    ensure_backtest_ready(df, price_col="close")
    pd.testing.assert_frame_equal(df, original)
