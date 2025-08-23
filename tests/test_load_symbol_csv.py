import pandas as pd
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
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 2


def test_load_symbol_csv_no_header_sorted(tmp_path):
    data_dir = tmp_path
    csv_path = data_dir / "EURUSD_H1.csv"
    csv_path.write_text(
        "\n".join(
            [
                "2020-01-01 01:00,1,2,0,1,200",
                "2020-01-01 00:00,1,2,0,1,100",
            ]
        )
    )

    df = load_symbol_csv("eurusd", data_dir=data_dir)
    expected = pd.to_datetime([
        "2020-01-01 00:00",
        "2020-01-01 01:00",
    ])
    assert df.index.equals(expected)
