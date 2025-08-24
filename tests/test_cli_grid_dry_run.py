from pathlib import Path
import ast
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


def test_cli_grid_dry_run(tmp_path, monkeypatch, capsys):
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

    called = False

    def fake_run_grid(*a, **k):
        nonlocal called
        called = True
        return pd.DataFrame()

    monkeypatch.setattr("forest5.cli.run_grid", fake_run_grid)

    rc = cmd_grid(args)
    assert rc == 0
    assert called is False

    out = capsys.readouterr().out.strip()
    assert out.startswith("dry-run")
    kw = ast.literal_eval(out.split("dry-run ", 1)[1])
    combos = len(kw["fast_values"]) * len(kw["slow_values"])
    assert combos == 4
