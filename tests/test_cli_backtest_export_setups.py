import pandas as pd
from pathlib import Path

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


def test_cli_backtest_export_setups(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    out_path = tmp_path / "setups.csv"
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
            "--export-setups",
            str(out_path),
        ]
    )

    rc = cmd_backtest(args)
    assert rc == 0
    assert out_path.exists()
    df = pd.read_csv(out_path)
    expected = {
        "time",
        "side",
        "entry",
        "sl",
        "tp",
        "drivers",
        "t_sep_atr",
        "pullback_atr",
        "rsi",
        "atr",
        "reasons",
    }
    assert expected.issubset(set(df.columns))
