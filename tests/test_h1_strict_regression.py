import pandas as pd
from pathlib import Path

from forest5.utils.io import normalize_ohlc_h1


def test_normalize_strict_from_padded(tmp_path: Path):
    idx = pd.date_range("2020-01-01 00:00", periods=5, freq="1h")
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [10, 10, 10, 10, 10],
        },
        index=idx,
    ).drop(idx[2])
    padded = normalize_ohlc_h1(df.copy(), policy="pad")
    assert pd.infer_freq(padded.index) == "h"
    strict = normalize_ohlc_h1(padded.copy(), policy="strict")
    assert pd.infer_freq(strict.index) == "h"
    assert not strict.isna().any().any()
