import json
from pathlib import Path

import pandas as pd

from forest5.cli import main


def _write_csv(path: Path, rows: int = 3) -> Path:
    idx = pd.date_range("2020-01-01", periods=rows, freq="h")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1.0] * rows,
            "high": [1.1] * rows,
            "low": [0.9] * rows,
            "close": [1.0] * rows,
        }
    )
    df.to_csv(path, index=False)
    return path


def test_data_inspect_input_dir(tmp_path: Path):
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    _write_csv(inp / "a.csv", rows=3)
    _write_csv(inp / "b.csv", rows=4)

    rc = main(["data", "inspect", "--input-dir", str(inp), "--out", str(out)])
    assert rc == 0

    a_txt = out / "a_summary.txt"
    a_js = out / "a_summary.json"
    b_txt = out / "b_summary.txt"
    b_js = out / "b_summary.json"

    assert a_txt.exists() and a_js.exists()
    assert b_txt.exists() and b_js.exists()

    a_info = json.loads(a_js.read_text())
    b_info = json.loads(b_js.read_text())
    assert a_info["rows"] == 3
    assert b_info["rows"] == 4

