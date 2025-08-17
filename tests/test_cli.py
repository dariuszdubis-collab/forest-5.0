import pandas as pd
import pytest

from forest5.cli import main


def _write_csv(path):
    idx = pd.date_range("2020-01-01", periods=5, freq="h")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1, 1.1, 1.2, 1.3, 1.4],
            "high": [1.1, 1.2, 1.3, 1.4, 1.5],
            "low": [0.9, 1.0, 1.1, 1.2, 1.3],
            "close": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_cli_backtest(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    monkeypatch.chdir(tmp_path)
    rc = main([
        "backtest",
        "--csv",
        str(csv_path),
        "--fast",
        "2",
        "--slow",
        "3",
        "--atr-period",
        "1",
    ])
    assert rc == 0


def test_cli_grid(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    monkeypatch.chdir(tmp_path)
    rc = main([
        "grid",
        "--csv",
        str(csv_path),
        "--fast-values",
        "2",
        "--slow-values",
        "3",
        "--jobs",
        "1",
        "--top",
        "1",
    ])
    assert rc == 0


def test_cli_missing_file():
    with pytest.raises(FileNotFoundError):
        main(["backtest", "--csv", "no_such_file.csv"])


def test_cli_missing_time_column(tmp_path):
    bad_csv = tmp_path / "bad_time.csv"
    bad_csv.write_text(
        "time,open,high,low,close\n" "bad,1,1,1,1\n" "also_bad,1,1,1,1\n"
    )
    with pytest.raises(ValueError, match="Failed to parse"):
        main(["backtest", "--csv", str(bad_csv), "--time-col", "time"])


def test_cli_missing_ohlc_columns(tmp_path):
    bad_csv = tmp_path / "bad_ohlc.csv"
    bad_csv.write_text("time,open,high,low\n2020-01-01,1,1,1\n")
    with pytest.raises(ValueError, match="CSV missing required columns"):
        main(["backtest", "--csv", str(bad_csv)])
