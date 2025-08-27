import sys
import importlib.util
from pathlib import Path

import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "optimize_grid", Path(__file__).resolve().parents[1] / "scripts" / "optimize_grid.py"
)
og = importlib.util.module_from_spec(SPEC)
sys.modules["optimize_grid"] = og
SPEC.loader.exec_module(og)  # type: ignore[attr-defined]


def _mk_csv(path):
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=5, freq="h"),
            "open": [1, 1, 1, 1, 1],
            "high": [1, 1, 1, 1, 1],
            "low": [1, 1, 1, 1, 1],
            "close": [1, 1, 1, 1, 1],
        }
    )
    df.to_csv(path, index=False)
    return path


calls = []


def _stub_run_one(df, gp, base, dd):
    calls.append((gp.fast, gp.slow))
    return og.GridResult(
        fast=gp.fast,
        slow=gp.slow,
        use_rsi=gp.use_rsi,
        rsi_period=gp.rsi_period,
        rsi_oversold=gp.rsi_oversold,
        rsi_overbought=gp.rsi_overbought,
        ret=0.0,
        max_dd=0.0,
        trades=0,
        equity_end=0.0,
        pnl_net=0.0,
        sharpe=0.0,
        score=0.0,
    )


def test_optimize_grid_aliases(tmp_path, monkeypatch):
    csv = _mk_csv(tmp_path / "data.csv")
    monkeypatch.setattr(og, "_run_one", _stub_run_one)
    monkeypatch.setattr(og, "_export_csv", lambda *a, **k: None)
    monkeypatch.setattr(og, "log_json", lambda **k: None)
    monkeypatch.setattr(og, "_print_top", lambda *a, **k: None)

    def run(args):
        monkeypatch.setattr(sys, "argv", ["optimize_grid.py"] + args)
        og.main()
    run(["--csv", str(csv), "--symbol", "EURUSD", "--fast", "1", "--slow", "2", "--jobs", "1"])
    run(["--csv", str(csv), "--symbol", "EURUSD", "--ema-fast", "1", "--ema-slow", "2", "--jobs", "1"])
    assert calls == [(1, 2), (1, 2)]
