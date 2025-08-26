import pandas as pd
from unittest.mock import patch, ANY

from forest5.backtest.engine import BacktestEngine
from forest5.config import BacktestSettings
from forest5.signals.contract import TechnicalSignal
from forest5.utils.log import E_SETUP_EXPIRE, R_TIMEOUT


def test_ttl_minutes_expires_over_weekend():
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-05 21:00",  # Friday
                "2024-01-08 00:00",  # Monday
            ]
        ),
    )
    settings = BacktestSettings(setup_ttl_minutes=60)
    eng = BacktestEngine(df, settings)

    sig = TechnicalSignal(action="BUY", entry=1.0, sl=0.0, tp=2.0)
    with patch.object(BacktestEngine, "_compute_signal_contract", return_value=sig):
        eng.on_bar_close(0)
    assert eng.setups._setups

    with patch("forest5.signals.setups.log_event") as le:
        eng.on_bar_open(1)

    le.assert_any_call(E_SETUP_EXPIRE, ctx=ANY, key="0", index=1, reason=R_TIMEOUT)
    assert not eng.setups._setups
    assert eng.positions == []
