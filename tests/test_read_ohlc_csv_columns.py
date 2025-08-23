import pandas as pd
import pytest

from forest5.cli import load_ohlc_csv
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


def test_read_ohlc_csv_handles_volume(tmp_path):
    df = pd.DataFrame(
        {
            "time": ["2020-01-01 00:00", "2020-01-01 01:00"],
            "Open": [1, 2],
            "High": [1, 2],
            "Low": [1, 2],
            "Close": [1, 2],
            "V": [1, "bad"],
        }
    )
    csv_path = tmp_path / "sample_vol.csv"
    df.to_csv(csv_path, index=False)
    out = read_ohlc_csv(csv_path)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    # The second row has an invalid volume and should be dropped
    assert len(out) == 1
    assert out["volume"].iloc[0] == 1


def test_read_ohlc_csv_without_header(tmp_path):
    csv_path = tmp_path / "EURUSD_H1.csv"
    csv_path.write_text(
        "\n".join(
            [
                "2020-01-01 00:00,1,2,0,1,100",
                "2020-01-01 01:00,1,2,0,1,200",
            ]
        )
    )

    out = read_ohlc_csv(csv_path, has_header=False)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert len(out) == 2


def test_load_ohlc_csv_rejects_non_hourly_index(tmp_path):
    df = pd.DataFrame(
        {
            "time": ["2020-01-01 00:00", "2020-01-01 00:30"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
        }
    )
    csv_path = tmp_path / "bad_freq.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="1H"):
        load_ohlc_csv(csv_path)
