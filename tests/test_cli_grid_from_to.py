from pathlib import Path

import pandas as pd

from forest5.cli import build_parser, cmd_grid


def _write_csv(path: Path, periods: int = 6) -> Path:
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


def test_cli_grid_respects_from_to(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv", periods=6)

    parser = build_parser()
    args = parser.parse_args(
        [
            "grid",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--fast-values",
            "1",
            "--slow-values",
            "2",
            "--from",
            "2020-01-01T02:00:00",
            "--to",
            "2020-01-01T04:00:00",
        ]
    )

    captured_df = {}

    def fake_run_grid(df, **kwargs):
        captured_df["df"] = df
        return pd.DataFrame(
            [{"fast": 1, "slow": 2, "equity_end": 0.0, "max_dd": 0.0, "cagr": 0.0, "rar": 0.0}]
        )

    monkeypatch.setattr("forest5.cli.run_grid", fake_run_grid)

    rc = cmd_grid(args)
    assert rc == 0

    df = captured_df["df"]
    assert list(df.index) == list(pd.date_range("2020-01-01T02:00:00", periods=3, freq="h"))
