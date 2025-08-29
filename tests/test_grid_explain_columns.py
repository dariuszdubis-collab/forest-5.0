import pandas as pd
from forest5.config import BacktestSettings
from forest5.grid.engine import plan_param_grid, run_grid
from forest5.examples.synthetic import generate_ohlc


def test_grid_explain_columns():
    df = generate_ohlc(periods=10, start_price=100.0, freq="H")
    settings = BacktestSettings(symbol="SYMB", timeframe="1h", strategy={"name": "ema_cross"})
    combos = plan_param_grid({"fast": [5], "slow": [10]})
    res = run_grid(df, combos, settings, jobs=0, explain=True)
    for col in [
        "cand_cnt",
        "gate_trend",
        "gate_pullback",
        "gate_rsi",
        "miss_pattern",
        "wait_timeonly",
        "ttl_expire",
    ]:
        assert col in res.columns
    sums = (
        res["gate_trend"]
        + res["gate_pullback"]
        + res["gate_rsi"]
        + res["miss_pattern"]
        + res["wait_timeonly"]
        + res["ttl_expire"]
    )
    assert (res["cand_cnt"] == sums).all()
