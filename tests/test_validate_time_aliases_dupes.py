import pandas as pd
from forest5.utils.validate import ensure_backtest_ready


def test_time_aliases_and_duplicates():
    # wejście z aliasem 'datetime' i duplikatem
    df = pd.DataFrame(
        {
            "datetime": ["2020-01-01 00:00:00", "2020-01-01 00:00:00", "2020-01-01 01:00:00"],
            "open": [1, 1, 1],
            "high": [1, 1, 1],
            "low": [1, 1, 1],
            "close": [1, 1, 1],
        }
    )
    out = ensure_backtest_ready(df, price_col="close")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing
    assert out.index.has_duplicates is False
