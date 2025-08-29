from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from forest5.cli import build_parser, cmd_backtest


def _write_csv(path: Path) -> Path:
    idx = pd.date_range("2020-01-01", periods=5, freq="h")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1, 1.1, 1.2, 1.3, 1.4],
            "high": [1.1, 1.2, 1.3, 1.4, 1.5],
            "low": [0.9, 1.0, 1.1, 1.2, 1.3],
            "close": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_backtest_passes_pattern_params(tmp_path, monkeypatch):
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
            "--pinbar-wick-dom",
            "0.6",
            "--star-reclaim-min",
            "0.62",
        ]
    )

    captured: dict[str, dict] = {}

    monkeypatch.setattr(
        "forest5.signals.patterns.registry.enable_engulfing",
        lambda **kw: captured.setdefault("engulf", kw),
    )
    monkeypatch.setattr(
        "forest5.signals.patterns.registry.enable_pinbar",
        lambda **kw: captured.setdefault("pinbar", kw),
    )
    monkeypatch.setattr(
        "forest5.signals.patterns.registry.enable_stars",
        lambda **kw: captured.setdefault("stars", kw),
    )

    def fake_run_backtest(df, settings, symbol, price_col, atr_period, atr_multiple):
        from forest5.signals.h1_ema_rsi_atr import compute_primary_signal_h1

        compute_primary_signal_h1(df, params=settings.strategy)
        return SimpleNamespace(
            equity_curve=pd.Series([1.0, 1.1]),
            max_dd=0.0,
            trades=SimpleNamespace(trades=[]),
        )

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    rc = cmd_backtest(args)
    assert rc == 0
