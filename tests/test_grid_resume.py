from pathlib import Path

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


def test_grid_resume(tmp_path, monkeypatch, capsys):
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
    assert cmd_grid(args) == 0

    # simulate partial results
    existing = pd.DataFrame(
        [
            {
                "combo_id": 0,
                "fast": 1,
                "slow": 3,
                "equity_end": 4.0,
                "dd": 0.1,
                "cagr": 0.2,
                "rar": 2.0,
            },
            {
                "combo_id": 1,
                "fast": 1,
                "slow": 4,
                "equity_end": 5.0,
                "dd": 0.1,
                "cagr": 0.2,
                "rar": 2.0,
            },
        ]
    )
    existing.to_csv(tmp_path / "results.csv", index=False)

    runs = []

    def fake_run_grid(df, fast_values, slow_values, **kwargs):
        runs.append((fast_values[0], slow_values[0]))
        return pd.DataFrame(
            [
                {
                    "fast": fast_values[0],
                    "slow": slow_values[0],
                    "risk": kwargs.get("risk_values", [kwargs.get("risk", 0.01)])[0],
                    "rsi_period": kwargs.get("rsi_period_values", [kwargs.get("rsi_period", 14)])[
                        0
                    ],
                    "max_dd": kwargs.get("max_dd_values", [kwargs.get("max_dd", 0.3)])[0],
                    "equity_end": fast_values[0] + slow_values[0],
                    "dd": 0.1,
                    "cagr": 0.2,
                    "rar": 2.0,
                }
            ]
        )

    monkeypatch.setattr("forest5.cli.run_grid", fake_run_grid)

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
            "--top",
            "1",
        ]
    )
    assert cmd_grid(args) == 0

    # verify skipped combos
    assert runs == [(2, 3), (2, 4)]

    results_path = tmp_path / "results.csv"
    assert results_path.exists()
    df_all = pd.read_csv(results_path)
    assert set(df_all["combo_id"]) == {0, 1, 2, 3}

    top_path = tmp_path / "results_top.csv"
    assert top_path.exists()
    df_top = pd.read_csv(top_path)
    assert len(df_top) == 1
    assert df_top.iloc[0]["combo_id"] == 3

    out_lines = capsys.readouterr().out.strip().splitlines()
    assert any("combo_id" in ln for ln in out_lines)
