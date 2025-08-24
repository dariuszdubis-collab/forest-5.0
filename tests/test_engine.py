import pandas as pd
from unittest.mock import patch

from forest5.backtest.engine import BacktestEngine, run_backtest
from forest5.config import BacktestSettings
from forest5.signals.setups import SetupCandidate


def _capture_atr(storage):
    def _inner(high, low, close, period):
        from forest5.core.indicators import atr as real_atr

        result = real_atr(high, low, close, period)
        storage[period] = result
        return result

    return _inner


def test_atr_period_override_changes_atr():
    idx = pd.date_range("2020", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [9.6, 10.6, 11.6, 12.6, 13.6],
            "high": [10, 11, 12, 13, 14],
            "low": [9, 10, 11, 12, 13],
            "close": [9.5, 10.5, 11.5, 12.5, 13.5],
        },
        index=idx,
    )
    settings = BacktestSettings()

    storage: dict[int, pd.Series] = {}
    with patch("forest5.backtest.engine.atr", side_effect=_capture_atr(storage)):
        run_backtest(df, settings)
    default_atr = storage[settings.atr_period]

    storage = {}
    with patch("forest5.backtest.engine.atr", side_effect=_capture_atr(storage)):
        run_backtest(df, settings, atr_period=1)
    override_atr = storage[1]

    assert not override_atr.equals(default_atr)


def test_gap_fill_trailing_and_priority():
    idx = pd.RangeIndex(3)
    df = pd.DataFrame(
        {
            "open": [100, 104, 103],
            "high": [100, 104, 103],
            "low": [100, 103.6, 100],
            "close": [100, 104, 102],
            "atr": [1.0, 1.0, 1.0],
        },
        index=idx,
    )
    settings = BacktestSettings(tp_sl_priority="TP_FIRST")
    eng = BacktestEngine(df, settings)
    cand = SetupCandidate(
        id="s1",
        action="BUY",
        entry=100.0,
        sl=0.0,
        tp=110.0,
        meta={"trailing_atr": 0.5},
    )
    eng.setups.arm("s1", 0, cand)

    eng.on_bar_open(1)
    assert eng.positions and eng.positions[0]["entry"] == 104.0

    eng.on_bar_close(1)
    assert eng.positions[0]["sl"] > 95.0

    eng.on_bar_open(2)
    eng.on_bar_close(2)
    assert not eng.positions
    assert eng.equity < 0.0
    assert eng.tp_sl_policy.priority == "TP_FIRST"
