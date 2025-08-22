import pandas as pd
from forest5.signals.candles import candles_signal
from forest5.signals.combine import confirm_with_candles


def _bullish_df():
    idx = pd.date_range("2023", periods=2, freq="h")
    return pd.DataFrame(
        {
            "open": [10, 8],
            "high": [10, 11],
            "low": [8, 7],
            "close": [9, 11],
        },
        index=idx,
    )


def _bearish_df():
    idx = pd.date_range("2023", periods=2, freq="h")
    return pd.DataFrame(
        {
            "open": [8, 10],
            "high": [9, 10],
            "low": [7, 6],
            "close": [9, 7],
        },
        index=idx,
    )


def _doji_df():
    idx = pd.date_range("2023", periods=2, freq="h")
    return pd.DataFrame(
        {
            "open": [10, 10],
            "high": [11, 11],
            "low": [9, 9],
            "close": [10, 10.1],
        },
        index=idx,
    )


def test_bullish_aligned_passes():
    df = _bullish_df()
    base = pd.Series([0, 1], index=df.index)
    candles = candles_signal(df)
    out = confirm_with_candles(base, candles)
    assert out.iloc[-1] == 1  # aligned -> pass through


def test_bearish_conflict_zeroed():
    df = _bearish_df()
    base = pd.Series([0, 1], index=df.index)
    candles = candles_signal(df)
    out = confirm_with_candles(base, candles)
    assert out.iloc[-1] == 0  # conflict -> neutralized


def test_doji_keeps_base():
    df = _doji_df()
    base = pd.Series([0, -1], index=df.index)
    candles = candles_signal(df)
    out = confirm_with_candles(base, candles)
    assert out.iloc[-1] == -1  # doji -> no change
