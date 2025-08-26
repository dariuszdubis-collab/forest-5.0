import pandas as pd
import pytest
from types import SimpleNamespace

from forest5.cli import build_parser, cmd_backtest, cmd_grid, cmd_walkforward


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


@pytest.mark.parametrize(
    "policy, expected_len, expected_ttl",
    [("pad", 3, None), ("drop", 2, 120)],
)
def test_backtest_h1_policy(tmp_path, monkeypatch, policy, expected_len, expected_ttl):
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
        captured["ttl"] = settings.setup_ttl_minutes
        return SimpleNamespace(
            equity_curve=pd.Series([1.0]),
            max_dd=0.0,
            trades=SimpleNamespace(trades=[]),
        )

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    cmd_backtest(args)

    assert captured["len"] == expected_len
    assert captured["ttl"] == expected_ttl


@pytest.mark.parametrize(
    "policy, expected_ttl",
    [("pad", None), ("drop", 120)],
)
def test_walkforward_h1_policy(tmp_path, monkeypatch, policy, expected_ttl):
    csv_path = _write_gap_csv(tmp_path / "data.csv")
    parser = build_parser()
    args = parser.parse_args(
        [
            "walkforward",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--h1-policy",
            policy,
            "--ema-fast",
            "2",
            "--ema-slow",
            "5",
            "--train",
            "1",
            "--test",
            "1",
        ]
    )

    captured = {}

    def fake_run_backtest(df, settings):
        captured["ttl"] = settings.setup_ttl_minutes
        return SimpleNamespace(
            equity_curve=pd.Series([1.0]),
            max_dd=0.0,
            trades=SimpleNamespace(trades=[]),
        )

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    cmd_walkforward(args)

    assert captured["ttl"] == expected_ttl


@pytest.mark.parametrize(
    "policy, expected_len, expected_ttl",
    [("pad", 3, None), ("drop", 2, 120)],
)
def test_grid_h1_policy(tmp_path, monkeypatch, policy, expected_len, expected_ttl):
    csv_path = _write_gap_csv(tmp_path / "data.csv")
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
            "--h1-policy",
            policy,
        ]
    )

    captured = {}

    def fake_run_grid(df, symbol, fast_values, slow_values, **kwargs):
        captured["len"] = len(df)
        captured["ttl"] = kwargs.get("setup_ttl_minutes")
        return pd.DataFrame([])

    monkeypatch.setattr("forest5.cli.run_grid", fake_run_grid)

    cmd_grid(args)

    assert captured["len"] == expected_len
    assert captured["ttl"] == expected_ttl
