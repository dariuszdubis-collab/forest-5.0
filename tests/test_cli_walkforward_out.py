from pathlib import Path

import pandas as pd

from forest5.cli import build_parser, cmd_walkforward


def _write_csv(path: Path) -> Path:
    idx = pd.date_range("2020-01-01", periods=6, freq="h")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "high": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "low": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            "close": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_walkforward_writes_out(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    out_path = tmp_path / "wf.csv"

    parser = build_parser()
    args = parser.parse_args(
        [
            "walkforward",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--train",
            "3",
            "--test",
            "1",
            "--ema-fast",
            "5",
            "--ema-slow",
            "8",
            "--rsi-len",
            "14",
            "--atr-len",
            "14",
            "--out",
            str(out_path),
        ]
    )

    def fake_run_backtest(df, settings):
        class R:
            equity_curve = pd.Series([1000.0] * max(1, len(df.index)), index=df.index)

        return R()

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    rc = cmd_walkforward(args)
    assert rc == 0
    assert out_path.exists()
    df_out = pd.read_csv(out_path)
    assert "train_start" in df_out.columns
    assert "test_end" in df_out.columns
    assert "equity_end" in df_out.columns
    assert len(df_out) >= 1
