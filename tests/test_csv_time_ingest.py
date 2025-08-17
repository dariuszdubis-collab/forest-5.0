import pandas as pd
from forest5.utils.validate import ensure_backtest_ready

def _mk_df_with(col):
    idx = pd.date_range("2020-01-01", periods=5, freq="h")
    df = pd.DataFrame({
        col: idx.astype(str),
        "open": [1,1,1,1,1],
        "high": [1,1,1,1,1],
        "low":  [1,1,1,1,1],
        "close":[1,1,1,1,1],
    })
    return df

def test_ingest_time_aliases():
    for c in ("time","date","datetime","timestamp"):
        df = _mk_df_with(c)
        out = ensure_backtest_ready(df, price_col="close")
        assert isinstance(out.index, pd.DatetimeIndex)
        assert len(out) == 5

