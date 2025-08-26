from pathlib import Path
import pandas as pd
import pytest

from forest5.cli import build_parser, cmd_grid


def _write_csv(path: Path) -> Path:
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


def test_cli_grid_additional_options(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    model_path = tmp_path / "model.onnx"
    model_path.touch()

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
            "--strategy",
            "macd_cross",
            "--risk-values",
            "0.01,0.02",
            "--max-dd-values",
            "0.2,0.3",
            "--use-rsi",
            "--rsi-period",
            "7",
            "--rsi-oversold",
            "10",
            "--rsi-overbought",
            "90",
            "--max-dd",
            "0.2",
            "--fee",
            "0.001",
            "--slippage",
            "0.0001",
            "--atr-period",
            "5",
            "--atr-multiple",
            "3",
            "--time-model",
            str(model_path),
            "--min-confluence",
            "2",
            "--jobs",
            "1",
            "--top",
            "1",
        ]
    )

    captured = {}

    def fake_run_grid(df, combos, settings, **kwargs):
        captured["combos"] = combos
        captured["settings"] = settings
        row = combos.iloc[0].to_dict()
        row.update({"equity_end": 1.0, "dd": 0.0, "cagr": 0.0, "rar": 0.0})
        return pd.DataFrame([row])

    monkeypatch.setattr("forest5.cli.run_grid", fake_run_grid)

    cmd_grid(args)

    combos = captured["combos"]
    settings = captured["settings"]
    assert set(combos["risk"]) == pytest.approx({0.01, 0.02})
    assert set(combos["max_dd"]) == pytest.approx({0.2, 0.3})
    assert settings.strategy.name == "macd_cross"
    assert settings.strategy.use_rsi is True
    assert settings.strategy.rsi_period == 7
    assert settings.strategy.rsi_oversold == 10
    assert settings.strategy.rsi_overbought == 90
    assert settings.risk.max_drawdown == pytest.approx(0.2)
    assert settings.risk.fee_perc == pytest.approx(0.001)
    assert settings.risk.slippage_perc == pytest.approx(0.0001)
    assert settings.atr_period == 5
    assert settings.atr_multiple == 3
    assert settings.time.model.path == model_path
    assert settings.time.fusion_min_confluence == pytest.approx(2.0)
