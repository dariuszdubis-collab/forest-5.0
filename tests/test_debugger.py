import json
from pathlib import Path

import pandas as pd

from forest5.backtest.engine import run_backtest
from forest5.backtest.risk import RiskManager
from forest5.config import BacktestSettings, BacktestTimeSettings, StrategySettings


class FakeRiskManager(RiskManager):
    """Risk manager that returns zero size on first call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def position_size(self, price: float, atr: float, atr_multiple: float) -> float:
        self.calls += 1
        if self.calls == 1:
            return 0.0
        return super().position_size(price, atr, atr_multiple)


def _make_df() -> pd.DataFrame:
    times = pd.date_range("2021-01-01", periods=5, freq="h")
    close = pd.Series([100, 95, 105, 90, 110], index=times)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        },
        index=times,
    )
    return df


def test_debug_logger(tmp_path: Path) -> None:
    df = _make_df()
    settings = BacktestSettings(
        strategy=StrategySettings(fast=2, slow=3),
        time=BacktestTimeSettings(blocked_hours=[3]),
        debug_dir=tmp_path,
    )
    risk = FakeRiskManager()
    run_backtest(df, settings, risk=risk)

    log_file = tmp_path / "decision_log.jsonl"
    assert log_file.exists()
    lines = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines()]

    assert any(e["event"] == "skip_candle" and e.get("reason") == "time_block" for e in lines)
    assert any(e["event"] == "signal_rejected" and e.get("reason") == "qty_zero" for e in lines)
    assert any(e["event"].startswith("position_") for e in lines)
