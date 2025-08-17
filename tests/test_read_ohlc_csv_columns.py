import pandas as pd
from forest5.utils.io import read_ohlc_csv


def test_read_ohlc_csv_normalizes_ohlc_columns(tmp_path):
    df = pd.DataFrame(
        {
            "TIME": ["2020-01-01 00:00", "2020-01-01 01:00"],
            " Open ": [1, 2],
            "HI": [1, 2],
            "l": [1, 2],
            "close_price": [1, 2],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    out = read_ohlc_csv(csv_path, time_col="TIME")
    assert list(out.columns) == ["open", "high", "low", "close"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "time"
