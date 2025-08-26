import pandas as pd

from forest5.cli import main


def test_data_inspect_smoke(tmp_path):
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


def test_data_normalize_smoke(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "raw.csv").write_text(
        "Date,Time,O,H,L,C\n" "2020-01-01,00:00,1,2,0,1\n",
        encoding="utf-8",
    )
    rc = main(["data", "normalize", "--input-dir", str(in_dir), "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "raw.csv").exists()


def test_data_pad_h1_smoke(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "pad.csv").write_text(
        "time,open,high,low,close\n" "2020-01-01 00:00,1,1,1,1\n" "2020-01-01 02:00,1,1,1,1\n",
        encoding="utf-8",
    )
    rc = main(["data", "pad-h1", "--input-dir", str(in_dir), "--out-dir", str(out_dir)])
    assert rc == 0
    df = pd.read_csv(out_dir / "pad.csv")
    assert len(df) == 3
