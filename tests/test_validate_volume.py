import pandas as pd
from forest5.utils.validate import ensure_backtest_ready


def test_volume_optional_and_preserved():
    df = pd.DataFrame(
        {
            "time": ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"],
            "open": [1, 1, 1],
            "high": [1, 1, 1],
            "low": [1, 1, 1],
            "close": [1, 1, 1],
            "volume": [100, "200", None],
            "extra": [10, 20, 30],
        }
    )
    out = ensure_backtest_ready(df)
    assert "volume" in out.columns
    assert "extra" in out.columns
    assert len(out) == 2
    assert pd.api.types.is_numeric_dtype(out["volume"])
