import pandas as pd
import pytest

from forest5.cli import load_ohlc_csv


def test_load_ohlc_csv_missing_high(tmp_path):
    df = pd.DataFrame(
        {
            "time": ["2020-01-01 00:00", "2020-01-01 01:00"],
            "open": [1, 2],
            # brak kolumny "high"
            "low": [1, 2],
            "close": [1, 2],
        }
    )
    csv_path = tmp_path / "missing_high.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match=r"CSV missing required columns: \['high'\]"):
        load_ohlc_csv(csv_path, time_col="time")
