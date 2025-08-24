from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from forest5.cli import build_parser, cmd_walkforward


def _write_csv(path: Path, periods: int = 5) -> Path:
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


def test_walkforward_smoke(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv", periods=5)
    out_csv = tmp_path / "wf.csv"

    parser = build_parser()
    args = parser.parse_args(
        [
            "walkforward",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--train",
            "2",
            "--test",
            "1",
            "--ema-fast",
            "5",
            "--ema-slow",
            "10",
            "--rsi-len",
            "14",
            "--atr-len",
            "14",
            "--out",
            str(out_csv),
        ]
    )

    counter = {"i": 0}

    def fake_run_backtest(df_local, settings):
        counter["i"] += 1
        eq_end = float(counter["i"])
        return SimpleNamespace(
            equity_curve=pd.Series([eq_end], index=df_local.index[:1]),
            max_dd=0.0,
            trades=SimpleNamespace(trades=[]),
        )

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    rc = cmd_walkforward(args)
    assert rc == 0

    out_df = pd.read_csv(out_csv)
    assert len(out_df) >= 2
    agg = out_df["equity_end"].mean()
    expected = (len(out_df) + 1) / 2.0
    assert agg == pytest.approx(expected)
