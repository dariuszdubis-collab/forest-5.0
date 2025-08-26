import pytest

from forest5.utils.io import read_ohlc_csv_smart, sniff_csv_dialect


def test_separator_detection(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "time;open;high;low;close\n" "2020-01-01 00:00;1.0;1.0;1.0;1.0\n",
        encoding="utf-8",
    )
    sep, decimal, has_header = sniff_csv_dialect(path)
    assert sep == ";"
    assert decimal == "."
    assert has_header is True


def test_numeric_parsing(tmp_path):
    path = tmp_path / "comma.csv"
    path.write_text(
        "time;open;high;low;close\n" "2020-01-01 00:00;1,23;1,23;1,23;1,23\n",
        encoding="utf-8",
    )
    df = read_ohlc_csv_smart(path)
    assert df["open"].iloc[0] == pytest.approx(1.23)
