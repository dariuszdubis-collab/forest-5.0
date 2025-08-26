from pathlib import Path
import json
import pandas as pd

from forest5.cli import build_parser, cmd_grid


def _write_csv(path: Path, periods: int = 3) -> Path:
    idx = pd.date_range("2020-01-01", periods=periods, freq="h")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1.0 + i * 0.1 for i in range(periods)],
            "high": [1.1 + i * 0.1 for i in range(periods)],
            "low": [0.9 + i * 0.1 for i in range(periods)],
            "close": [1.0 + i * 0.1 for i in range(periods)],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_grid_dryrun_plan(tmp_path, monkeypatch, capsys):
    csv_path = _write_csv(tmp_path / "data.csv")
    parser = build_parser()
    args = parser.parse_args(
        [
            "grid",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--fast-values",
            "1,2",
            "--slow-values",
            "3,4",
            "--dry-run",
        ]
    )
    monkeypatch.chdir(tmp_path)
    rc = cmd_grid(args)
    assert rc == 0
    out_lines = capsys.readouterr().out.strip().splitlines()
    assert out_lines[-1] == "4"
    plan_path = tmp_path / "plan.csv"
    meta_path = tmp_path / "meta.json"
    assert plan_path.exists()
    assert meta_path.exists()
    df_plan = pd.read_csv(plan_path)
    assert list(df_plan.columns)[0] == "combo_id"
    assert len(df_plan) == 4
    meta = json.loads(meta_path.read_text())
    assert meta["total_combos"] == 4
