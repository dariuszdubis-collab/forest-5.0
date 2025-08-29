from pathlib import Path
import pandas as pd

from forest5.cli import build_parser, cmd_grid


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


def test_grid_range_parsing(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv")
    parser = build_parser()
    args = parser.parse_args(
        [
            "grid",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--ema-fast",
            "1-3",
            "--ema-slow",
            "4,5",
            "--rsi-len",
            "7:9:1",
            "--atr-len",
            "10",
            "--t-sep-atr",
            "0.5-1.0:0.25",
            "--pullback-atr",
            "0.5",
            "--entry-buffer-atr",
            "0.1",
            "--sl-min-atr",
            "0.2",
            "--rr",
            "1.5",
            "--trailing-atr",
            "0.5",
            "--q-low",
            "0.1",
            "--q-high",
            "0.9",
        ]
    )
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


def test_grid_plan_includes_pattern_columns(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "data.csv")
    parser = build_parser()
    args = parser.parse_args(
        [
            "grid",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--ema-fast",
            "1",
            "--ema-slow",
            "2",
            "--engulf-eps-atr",
            "0.03,0.05",
            "--engulf-body-ratio-min",
            "1.0,1.1",
            "--pinbar-wick-dom",
            "0.55,0.60",
            "--pinbar-body-max",
            "0.25,0.30",
            "--pinbar-opp-wick-max",
            "0.15,0.20",
            "--star-reclaim-min",
            "0.50,0.62",
            "--star-mid-small-max",
            "0.30,0.40",
            "--no-engulf",
            "--no-pinbar",
            "--no-star",
            "--dry-run",
            "--out",
            str(tmp_path),
        ]
    )

    monkeypatch.chdir(tmp_path)
    cmd_grid(args)
    plan = pd.read_csv(tmp_path / "plan.csv")
    expected = {
        "engulf_eps_atr",
        "engulf_body_ratio_min",
        "pinbar_wick_dom",
        "pinbar_body_max",
        "pinbar_opp_wick_max",
        "star_reclaim_min",
        "star_mid_small_max",
        "enable_engulf",
        "enable_pinbar",
        "enable_star",
    }
    assert expected.issubset(set(plan.columns))


def test_grid_fast_slow_alias(tmp_path):
    csv_path = _write_csv(tmp_path / "data.csv")
    parser = build_parser()
    args = parser.parse_args(
        [
            "grid",
            "--csv",
            str(csv_path),
            "--symbol",
            "EURUSD",
            "--fast",
            "12",
            "--slow",
            "24",
        ]
    )
    assert args.fast_values == [12]
    assert args.slow_values == [24]
