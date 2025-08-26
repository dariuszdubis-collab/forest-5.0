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
            "--resume",
            "true",
            "--chunks",
            "3",
            "--chunk-id",
            "2",
        ]
    )

    captured = {}

    def fake_run_grid(df, symbol, fast_values, slow_values, **kwargs):
        captured["kwargs"] = kwargs
        combos = pd.DataFrame(
            [
                {
                    "fast": 2,
                    "slow": 5,
                    "equity_end": 1.0,
                    "risk": 0.01,
                    "max_dd": 0.2,
                    "cagr": 0.0,
                    "rar": 0.0,
                },
                {
                    "fast": 2,
                    "slow": 5,
                    "equity_end": 1.0,
                    "risk": 0.02,
                    "max_dd": 0.3,
                    "cagr": 0.0,
                    "rar": 0.0,
                },
            ]
        )
        captured["combos"] = combos
        return combos

    monkeypatch.setattr("forest5.cli.run_grid", fake_run_grid)

    cmd_grid(args)

    kw = captured["kwargs"]
    combos = captured["combos"]
    assert kw["strategy"] == "macd_cross"
    # risk options should be unique
    assert sorted(combos["risk"].unique()) == pytest.approx([0.01, 0.02])
    # maximum drawdown options should be unique
    assert sorted(combos["max_dd"].unique()) == pytest.approx([0.2, 0.3])
    assert kw["use_rsi"] is True
    assert kw["rsi_period"] == 7
    assert kw["rsi_oversold"] == 10
    assert kw["rsi_overbought"] == 90
    assert kw["max_dd"] == pytest.approx(0.2)
    assert kw["fee"] == pytest.approx(0.001)
    assert kw["slippage"] == pytest.approx(0.0001)
    assert kw["atr_period"] == 5
    assert kw["atr_multiple"] == 3
    assert kw["time_model"] == model_path
    assert kw["min_confluence"] == pytest.approx(2.0)
    assert kw["resume"] is True
    assert kw["chunks"] == 3
    assert kw["chunk_id"] == 2
