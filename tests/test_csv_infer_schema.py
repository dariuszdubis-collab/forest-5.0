import pytest

from forest5.utils.io import read_ohlc_csv_smart


def test_headerless_csv(tmp_path):
    path = tmp_path / "no_header.csv"
    path.write_text(
        "2020-01-01 00:00,1,1,1,1,10\n" "2020-01-01 01:00,1,1,1,1,11\n",
        encoding="utf-8",
    )
    df = read_ohlc_csv_smart(path)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_column_aliases(tmp_path):
    path = tmp_path / "aliases.csv"
    path.write_text(
        "Date,Time,O,H,L,C,V\n" "2020-01-01,00:00,1,2,0,1,5\n" "2020-01-01,01:00,1,2,0,1,6\n",
        encoding="utf-8",
    )
    df = read_ohlc_csv_smart(path)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_decimal_commas(tmp_path):
    path = tmp_path / "commas.csv"
    path.write_text(
        "time;open;high;low;close\n" "2020-01-01 00:00;1,1;2,2;0,9;1,0\n",
        encoding="utf-8",
    )
    df = read_ohlc_csv_smart(path)
    assert df["open"].iloc[0] == pytest.approx(1.1)
    assert df["high"].iloc[0] == pytest.approx(2.2)
