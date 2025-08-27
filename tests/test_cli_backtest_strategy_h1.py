import pandas as pd
from types import SimpleNamespace

from forest5.cli import build_parser, cmd_backtest


def _write_csv(path):
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


def test_cli_backtest_h1_strategy(tmp_path, monkeypatch, capsys):
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
            "--ema-fast",
            "12",
            "--ema-slow",
            "34",
            "--rsi-len",
            "14",
            "--atr-len",
            "14",
            "--t-sep-atr",
            "0.2",
            "--pullback-atr",
            "0.5",
            "--entry-buffer-atr",
            "0.1",
            "--sl-min-atr",
            "0.9",
            "--rr",
            "1.8",
        ]
    )

    captured = {}

    def fake_run_backtest(df, settings, symbol, price_col, atr_period, atr_multiple):
        captured["settings"] = settings
        return SimpleNamespace(
            equity_curve=pd.Series([1.0, 1.1]),
            max_dd=0.0,
            trades=SimpleNamespace(trades=[]),
        )

    monkeypatch.setattr("forest5.cli.run_backtest", fake_run_backtest)

    rc = cmd_backtest(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Equity end" in out

    strat = captured["settings"].strategy
    assert strat.name == "h1_ema_rsi_atr"
    assert strat.params.ema_fast == 12
    assert strat.params.ema_slow == 34
    assert captured["settings"].time.q_low == 0.1
    assert captured["settings"].time.q_high == 0.9
