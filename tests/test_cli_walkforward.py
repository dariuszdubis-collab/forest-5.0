from pathlib import Path

import pandas as pd

from forest5.cli import build_parser, cmd_walkforward


def _write_csv(path: Path) -> Path:
    idx = pd.date_range("2020-01-01", periods=3, freq="h")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.0, 1.1, 1.2],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_cli_walkforward_basic(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")

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
        ]
    )

    captured = {}

    def fake_run_backtest(df, settings):
        captured["settings"] = settings

        class R:
            equity_curve = pd.Series([1.0], index=df.index)

        return R()

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    rc = cmd_walkforward(args)
    assert rc == 0
    st = captured["settings"]
    assert st.strategy.name == "h1_ema_rsi_atr"
    assert st.strategy.params.ema_fast == 5
    assert st.strategy.params.ema_slow == 10
