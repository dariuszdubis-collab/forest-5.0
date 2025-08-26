import pandas as pd
import pytest
from types import SimpleNamespace

from forest5.cli import build_parser, cmd_backtest


def _write_gap_csv(path):
    idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 02:00"])
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1.0, 1.2],
            "high": [1.1, 1.3],
            "low": [0.9, 1.1],
            "close": [1.0, 1.2],
        }
    )
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("policy, expected", [("pad", 3), ("drop", 2)])
def test_h1_policy_pad_drop(tmp_path, monkeypatch, policy, expected):
    csv_path = _write_gap_csv(tmp_path / "data.csv")
    parser = build_parser()
    args = parser.parse_args(
        [
            "backtest",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--h1-policy",
            policy,
        ]
    )

    captured = {}

    def fake_run_backtest(df, settings, symbol, price_col, atr_period, atr_multiple):
        captured["len"] = len(df)
        return SimpleNamespace(
            equity_curve=pd.Series([1.0]),
            max_dd=0.0,
            trades=SimpleNamespace(trades=[]),
        )

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    cmd_backtest(args)

    assert captured["len"] == expected
