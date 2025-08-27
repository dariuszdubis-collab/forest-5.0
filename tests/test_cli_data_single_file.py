import pandas as pd
from forest5.cli import main


def test_data_inspect_single_file(tmp_path):
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
    out_dir = tmp_path / "out"
    rc = main(["data", "inspect", "--csv", str(csv), "--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "data.txt").exists()
    assert (out_dir / "data.json").exists()


def test_data_pad_h1_single_file(tmp_path):
    csv = tmp_path / "pad.csv"
    csv.write_text(
        "time,open,high,low,close\n" "2020-01-01 00:00,1,1,1,1\n" "2020-01-01 02:00,1,1,1,1\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "out.csv"
    rc = main(["data", "pad-h1", "--csv", str(csv), "--policy", "strict", "--out", str(out_csv)])
    assert rc != 0
    rc = main(["data", "pad-h1", "--csv", str(csv), "--policy", "pad", "--out", str(out_csv)])
    assert rc == 0
    df = pd.read_csv(out_csv)
    assert len(df) == 3
    assert "2020-01-01 01:00:00" in df["time"].tolist()
