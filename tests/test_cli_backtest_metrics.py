import json
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from forest5.cli import build_parser, cmd_backtest


def _write_csv(path: Path) -> Path:
    idx = pd.date_range("2020-01-01", periods=5, freq="h")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1.0, 1.1, 1.2, 1.3, 1.4],
            "high": [1.1, 1.2, 1.3, 1.4, 1.5],
            "low": [0.9, 1.0, 1.1, 1.2, 1.3],
            "close": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_backtest_metrics_out(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    out_path = tmp_path / "metrics.json"

    parser = build_parser()
    args = parser.parse_args(
        [
            "backtest",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--fast",
            "2",
            "--slow",
            "3",
            "--atr-period",
            "1",
            "--metrics-out",
            str(out_path),
        ]
    )

    def fake_run_backtest(df, settings, symbol, price_col, atr_period, atr_multiple, collector=None):
        return SimpleNamespace(
            equity_curve=pd.Series([1000.0, 1010.0]),
            max_dd=0.05,
            trades=SimpleNamespace(trades=[]),
        )

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    rc = cmd_backtest(args)
    assert rc == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["symbol"] == "EURUSD"
    assert "equity_end" in data and data["equity_end"] == 1010.0
    assert "return" in data and isinstance(data["return"], float)
    assert data["trades"] == 0

