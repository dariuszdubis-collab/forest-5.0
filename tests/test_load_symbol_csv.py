import pandas as pd
import pytest
from forest5.utils.io import load_symbol_csv


def test_load_symbol_csv_no_header(tmp_path):
    data_dir = tmp_path
    csv_path = data_dir / "EURUSD_H1.csv"
    csv_path.write_text(
        "\n".join(
            [
                "2020-01-01 00:00,1,2,0,1,100",
                "2020-01-01 01:00,1,2,0,1,200",
            ]
        )
    )

    df = load_symbol_csv("eurusd", data_dir=data_dir)
    expected = pd.to_datetime(
        [
            "2020-01-01 00:00",
            "2020-01-01 01:00",
        ]
    )
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.equals(expected)
    assert len(df) == 2


def test_load_symbol_csv_rejects_unknown_symbol(tmp_path):
    with pytest.raises(ValueError):
        load_symbol_csv("FOOBAR", data_dir=tmp_path)
