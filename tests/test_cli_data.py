import pandas as pd

from forest5.cli import main


def test_data_inspect(tmp_path, capsys):
    csv = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=2, freq="h"),
            "open": [1, 1.1],
            "high": [1, 1.1],
            "low": [1, 1.1],
            "close": [1, 1.1],
        }
    )
    df.to_csv(csv, index=False)

    rc = main(["data", "inspect", "--csv", str(csv)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dialect" in out
    assert "date range" in out


def test_data_normalize(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    raw = in_dir / "raw.csv"
    raw.write_text(
        "Date,Time,O,H,L,C\n" "2020-01-01,00:00,1,2,0,1\n" "2020-01-01,01:00,1,2,0,1\n",
        encoding="utf-8",
    )

    rc = main(["data", "normalize", "--input-dir", str(in_dir), "--out-dir", str(out_dir)])
    assert rc == 0
    df = pd.read_csv(out_dir / "raw.csv")
    assert list(df.columns) == ["time", "open", "high", "low", "close"]


def test_data_pad_h1(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    pad = in_dir / "pad.csv"
    pad.write_text(
        "time,open,high,low,close\n"
        "2020-01-01 00:00,1,1,1,1\n"
        "2020-01-01 02:00,1,1,1,1\n",
        encoding="utf-8",
    )

    rc = main(["data", "pad-h1", "--input-dir", str(in_dir), "--out-dir", str(out_dir)])
    assert rc == 0
    df = pd.read_csv(out_dir / "pad.csv")
    assert len(df) == 3
    assert "2020-01-01 01:00:00" in df["time"].tolist()

