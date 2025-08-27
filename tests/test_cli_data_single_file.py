import json

import pandas as pd

from forest5.cli import main


def _write_csv(path):
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=3, freq="h"),
            "open": [1.0, 1.1, 1.2],
            "high": [1.0, 1.1, 1.2],
            "low": [1.0, 1.1, 1.2],
            "close": [1.0, 1.1, 1.2],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_data_inspect_single_file(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv")
    out_dir = tmp_path / "out"
    rc = main(["data", "inspect", "--csv", str(csv_path), "--out", str(out_dir)])
    assert rc == 0
    txt = out_dir / "summary.txt"
    js = out_dir / "summary.json"
    assert txt.exists()
    assert js.exists()
    info = json.loads(js.read_text())
    assert info["rows"] == 3


def test_data_pad_h1_single_file(tmp_path):
    gappy = tmp_path / "gappy.csv"
    gappy.write_text(
        "time,open,high,low,close\n" "2020-01-01 00:00,1,1,1,1\n" "2020-01-01 02:00,1,1,1,1\n",
        encoding="utf-8",
    )
    out_file = tmp_path / "filled.csv"

    rc = main(
        [
            "data",
            "pad-h1",
            "--csv",
            str(gappy),
            "--policy",
            "strict",
            "--out",
            str(out_file),
        ]
    )
    assert rc == 1

    rc = main(
        [
            "data",
            "pad-h1",
            "--csv",
            str(gappy),
            "--policy",
            "pad",
            "--out",
            str(out_file),
        ]
    )
    assert rc == 0
    df = pd.read_csv(out_file)
    assert len(df) == 3
