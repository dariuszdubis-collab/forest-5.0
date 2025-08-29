from pathlib import Path
import numbers
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
            "0",
            "--top",
            "1",
            "--out",
            str(tmp_path),
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

    # compare numeric parameters via sorted lists for determinism
    risk_vals = sorted(combos["risk"].unique())
    if all(isinstance(x, numbers.Real) for x in risk_vals):
        assert risk_vals == pytest.approx([0.01, 0.02])
    else:
        assert risk_vals == [0.01, 0.02]

    max_dd_vals = sorted(combos["max_dd"].unique())
    if all(isinstance(x, numbers.Real) for x in max_dd_vals):
        assert max_dd_vals == pytest.approx([0.2, 0.3])
    else:
        assert max_dd_vals == [0.2, 0.3]
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


def test_pattern_flags_parse(tmp_path):
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
            "2",
            "--slow-values",
            "5",
            "--engulf-eps-atr",
            "0.03,0.05",
            "--engulf-body-ratio-min",
            "1.0,1.1",
            "--pinbar-wick-dom",
            "0.55,0.60",
            "--pinbar-body-max",
            "0.25,0.30",
            "--pinbar-opp-wick-max",
            "0.15,0.20",
            "--star-reclaim-min",
            "0.50,0.62",
            "--star-mid-small-max",
            "0.30,0.40",
            "--no-star",
        ]
    )
    assert args.engulf_eps_atr == [0.03, 0.05]
    assert args.engulf_body_ratio_min == [1.0, 1.1]
    assert args.pinbar_wick_dom == [0.55, 0.60]
    assert args.pinbar_body_max == [0.25, 0.30]
    assert args.pinbar_opp_wick_max == [0.15, 0.20]
    assert args.star_reclaim_min == [0.50, 0.62]
    assert args.star_mid_small_max == [0.30, 0.40]
    assert args.no_star is True


def test_backtest_pattern_flags_parse(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv")
    parser = build_parser()
    args = parser.parse_args(
        [
            "backtest",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--strategy",
            "h1_ema_rsi_atr",
            "--engulf-eps-atr",
            "0.05",
            "--engulf-body-ratio-min",
            "1.1",
            "--pinbar-wick-dom",
            "0.6",
            "--pinbar-body-max",
            "0.3",
            "--pinbar-opp-wick-max",
            "0.2",
            "--star-reclaim-min",
            "0.62",
            "--star-mid-small-max",
            "0.4",
            "--no-engulf",
        ]
    )
    assert args.engulf_eps_atr == pytest.approx(0.05)
    assert args.pinbar_wick_dom == pytest.approx(0.6)
    assert args.star_mid_small_max == pytest.approx(0.4)
    assert args.no_engulf is True
