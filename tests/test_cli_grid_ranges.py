from pathlib import Path
import pandas as pd

from forest5.cli import build_parser


def _write_csv(path: Path) -> Path:
    idx = pd.date_range("2020-01-01", periods=3, freq="h")
    df = pd.DataFrame({
        "time": idx,
        "open": [1.0, 1.1, 1.2],
        "high": [1.1, 1.2, 1.3],
        "low": [0.9, 1.0, 1.1],
        "close": [1.0, 1.1, 1.2],
    })
    df.to_csv(path, index=False)
    return path


def test_grid_range_parsing(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv")
    parser = build_parser()
    args = parser.parse_args([
        "grid",
        "--csv", str(csv_path),
        "--symbol", "EURUSD",
        "--ema-fast", "1-3",
        "--ema-slow", "4,5",
        "--rsi-len", "7:9:1",
        "--atr-len", "10",
        "--t-sep-atr", "0.5-1.0:0.25",
        "--pullback-atr", "0.5",
        "--entry-buffer-atr", "0.1",
        "--sl-min-atr", "0.2",
        "--rr", "1.5",
        "--trailing-atr", "0.5",
        "--q-low", "0.1",
        "--q-high", "0.9",
    ])
    assert args.fast_values == [1, 2, 3]
    assert args.slow_values == [4, 5]
    assert args.rsi_period == [7, 8, 9]
    assert args.atr_period == [10]
    assert args.t_sep_atr == [0.5, 0.75, 1.0]
    assert args.pullback_atr == [0.5]
    assert args.entry_buffer_atr == [0.1]
    assert args.sl_min_atr == [0.2]
    assert args.rr == [1.5]
    assert args.trailing_atr == [0.5]
    assert args.q_low == [0.1]
    assert args.q_high == [0.9]


def test_grid_fast_slow_alias(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv")
    parser = build_parser()
    args = parser.parse_args([
        "grid",
        "--csv", str(csv_path),
        "--symbol", "EURUSD",
        "--fast", "12",
        "--slow", "24",
    ])
    assert args.fast_values == [12]
    assert args.slow_values == [24]
