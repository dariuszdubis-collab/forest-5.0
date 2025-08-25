import pandas as pd
from unittest.mock import patch

from forest5.backtest.engine import BacktestEngine
from forest5.config import BacktestSettings
from forest5.signals.setups import TriggeredSignal
from forest5.utils.log import E_ORDER_REJECTED


def _make_engine() -> BacktestEngine:
    df = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        }
    )
    settings = BacktestSettings()
    return BacktestEngine(df, settings)


def test_open_order_rejected_logs_event():
    eng = _make_engine()
    sig = TriggeredSignal(
        setup_id="s1",
        action="BUY",
        entry=1.0,
        sl=0.0,
        tp=2.0,
        meta={"qty": 0.0},
    )
    with patch("forest5.backtest.engine.log_event") as le:
        eng._open_position(sig, entry=1.0, index=0)
    le.assert_called_once()
    event, ctx = le.call_args[0][:2]
    kwargs = le.call_args[1]
    assert event == E_ORDER_REJECTED
    assert kwargs["reason"] == "qty_zero"
    assert kwargs["client_order_id"]
    assert ctx.setup_id == "s1"


def test_close_order_rejected_logs_event():
    eng = _make_engine()
    pos = {
        "id": "s1",
        "action": "BUY",
        "entry": 1.0,
        "sl": 0.0,
        "tp": 2.0,
        "orig_sl": 0.0,
        "orig_tp": 2.0,
        "open_index": 0,
        "horizon": 0,
        "meta": {},
        "ticket": 1,
        "client_order_id": "cid1",
        "qty": 0.0,
    }
    with patch("forest5.backtest.engine.log_event") as le:
        eng._close_position(pos, price=1.0)
    le.assert_called_once()
    event, ctx = le.call_args[0][:2]
    kwargs = le.call_args[1]
    assert event == E_ORDER_REJECTED
    assert kwargs["reason"] == "qty_zero"
    assert kwargs["client_order_id"]
    assert ctx.setup_id == "s1"
