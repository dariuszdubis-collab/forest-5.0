import pandas as pd

from forest5.backtest.engine import BacktestEngine
from forest5.config import BacktestSettings
from forest5.signals.setups import SetupCandidate


def test_contract_sl_tp_preserved() -> None:
    """Positions opened from contract keep their SL/TP even with trailing."""

    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [104.0, 103.0],
            "low": [99.0, 100.0],
            "close": [102.0, 103.0],
            "atr": [1.0, 1.0],
        }
    )

    settings = BacktestSettings()
    engine = BacktestEngine(df, settings)

    cand = SetupCandidate(
        id="s1",
        action="BUY",
        entry=101.0,
        sl=99.0,
        tp=105.0,
        meta={"trailing_atr": 1.0},
    )

    engine._open_position(cand, entry=cand.entry, index=1)

    assert engine.positions[0]["sl"] == 99.0
    assert engine.positions[0]["tp"] == 105.0

    engine._update_open_positions(1, df.iloc[1])

    pos = engine.positions[0]
    assert pos["sl"] == 99.0
    assert pos["tp"] == 105.0
