import re
from pathlib import Path

import pandas as pd

from forest5.cli import build_parser, cmd_grid
from forest5.examples.synthetic import generate_ohlc
from forest5.grid.engine import plan_param_grid


def test_grid_resume(tmp_path, capsys):
    df = generate_ohlc(periods=10, start_price=100.0, freq="h")
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=True, date_format="%Y-%m-%d %H:%M:%S")

    parser = build_parser()
    results_path = tmp_path / "results.csv"

    args1 = parser.parse_args(
        [
            "grid",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--fast-values",
            "1",
            "--slow-values",
            "2,3",
            "--out",
            str(results_path),
            "--top",
            "2",
        ]
    )
    cmd_grid(args1)
    capsys.readouterr()  # clear

    args2 = parser.parse_args(
        [
            "grid",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--fast-values",
            "1,2",
            "--slow-values",
            "2,3",
            "--out",
            str(results_path),
            "--resume",
            "true",
            "--top",
            "2",
        ]
    )
    cmd_grid(args2)
    out = capsys.readouterr().out.lower()

    results = pd.read_csv(results_path)
    expected_ids = set(
        plan_param_grid(
            {"fast": [1, 2], "slow": [2, 3]}, filter_fn=lambda c: c["fast"] < c["slow"]
        )["combo_id"]
    )
    assert set(results["combo_id"]) == expected_ids
    assert results["combo_id"].is_unique

    assert (tmp_path / "results_top.csv").exists()
    assert re.search(r"skipp\w*\s+2", out)
