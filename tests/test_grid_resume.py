import json
from pathlib import Path

import pandas as pd

from forest5.cli import build_parser, cmd_grid
from forest5.examples.synthetic import generate_ohlc


def test_grid_resume(tmp_path, capsys):
    df = generate_ohlc(periods=50, start_price=100.0, freq="H")
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index_label="time")

    parser = build_parser()
    base = [
        "grid",
        "--csv",
        str(csv_path),
        "--symbol",
        "SYMB",
        "--fast-values",
        "21,34",
        "--slow-values",
        "89,144",
        "--out",
        str(tmp_path),
        "--seed",
        "1",
    ]

    args1 = parser.parse_args(base + ["--chunks", "2", "--chunk-id", "1"])
    rc1 = cmd_grid(args1)
    assert rc1 == 0
    res_path = tmp_path / "results.csv"
    assert res_path.exists()
    assert len(pd.read_csv(res_path)) == 2

    capsys.readouterr()
    args2 = parser.parse_args(base + ["--chunks", "2", "--chunk-id", "2", "--resume", "auto"])
    rc2 = cmd_grid(args2)
    assert rc2 == 0
    out = capsys.readouterr().out
    assert "skipping 2 of 4" in out

    merged = pd.read_csv(res_path)
    assert merged["combo_id"].nunique() == 4

    top_path = tmp_path / "results_top.csv"
    assert top_path.exists()
    top_df = pd.read_csv(top_path)
    assert len(top_df) <= int(args2.top or 20)

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["seed"] == 1
    assert meta["total_combos"] == 4
    assert meta["completed_combos"] == 4
