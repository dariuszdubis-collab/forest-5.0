import pandas as pd
from types import SimpleNamespace
import pytest

from forest5.cli import build_parser, cmd_backtest


def _write_csv(path):
    idx = pd.date_range("2020-01-01", periods=3, freq="h")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.0, 1.1, 1.2],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_cli_time_only_options(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    model_path = tmp_path / "time_model.onnx"
    model_path.touch()

    parser = build_parser()
    args = parser.parse_args(
        [
            "backtest",
            "--csv",
            str(csv_path),
            "--time-model",
            str(model_path),
            "--blocked-hours",
            "1,2",
            "--blocked-weekdays",
            "0,6",
        ]
    )

    captured = {}

    def fake_run_backtest(df, settings, symbol, price_col, atr_period, atr_multiple):
        captured["settings"] = settings
        return SimpleNamespace(
            equity_curve=pd.Series([1.0]),
            max_dd=0.0,
            trades=SimpleNamespace(trades=[]),
        )

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    cmd_backtest(args)

    settings = captured["settings"]
    assert settings.time.model.enabled is True
    assert settings.time.model.path == model_path
    assert settings.time.blocked_hours == [1, 2]
    assert settings.time.blocked_weekdays == [0, 6]


def test_cli_time_model_missing(tmp_path, capsys):
    csv_path = _write_csv(tmp_path / "data.csv")
    model_path = tmp_path / "missing.onnx"

    parser = build_parser()
    args = parser.parse_args(
        ["backtest", "--csv", str(csv_path), "--time-model", str(model_path)]
    )

    with pytest.raises(SystemExit) as exc:
        cmd_backtest(args)
    assert exc.value.code == 1

    out = capsys.readouterr().out
    assert f"Plik modelu czasu nie istnieje: {model_path}" in out
