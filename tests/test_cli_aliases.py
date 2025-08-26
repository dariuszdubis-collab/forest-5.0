import pandas as pd
from pathlib import Path
import pytest

from forest5.cli import build_parser, cmd_grid, cmd_walkforward


def _write_csv(path: Path) -> Path:
    idx = pd.date_range("2020-01-01", periods=4, freq="h")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.0, 1.1, 1.2, 1.3],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_cli_grid_pullback_alias(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")

    captured = {}

    def fake_run_grid(df, combos, settings, **kwargs):
        captured["pullback_atr"] = settings.strategy.pullback_atr
        row = combos.iloc[0].to_dict()
        row.update({"equity_end": 1.0, "dd": 0.0, "cagr": 0.0, "rar": 0.0})
        return pd.DataFrame([row])

    monkeypatch.setattr("forest5.cli.run_grid", fake_run_grid)

    parser = build_parser()
    args = parser.parse_args(
        [
            "grid",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--fast-values",
            "2",
            "--slow-values",
            "5",
            "--pullback-to-ema-fast-atr",
            "0.7",
        ]
    )

    cmd_grid(args)
    assert captured["pullback_atr"] == pytest.approx(0.7)


def test_cli_walkforward_pullback_alias(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")

    captured = {}

    def fake_run_backtest(df, settings, **kwargs):
        captured["pullback_atr"] = settings.strategy.params.pullback_atr

        class Res:
            equity_curve = pd.Series([1.0, 1.0])

        return Res()

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    parser = build_parser()
    args = parser.parse_args(
        [
            "walkforward",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--train",
            "1",
            "--test",
            "1",
            "--ema-fast",
            "21",
            "--ema-slow",
            "55",
            "--pullback-to-ema-fast-atr",
            "0.7",
        ]
    )

    cmd_walkforward(args)
    assert captured["pullback_atr"] == pytest.approx(0.7)
