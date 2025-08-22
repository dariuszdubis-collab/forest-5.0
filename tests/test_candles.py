import pandas as pd

from forest5.signals.candles import bullish_engulfing, bearish_engulfing, doji


def test_bullish_engulfing():
    df = pd.DataFrame(
        {
            "open": [10, 7],
            "close": [8, 11],
        }
    )
    assert bullish_engulfing(df).tolist() == [0, 1]


def test_bearish_engulfing():
    df = pd.DataFrame(
        {
            "open": [8, 11],
            "close": [10, 7],
        }
    )
    assert bearish_engulfing(df).tolist() == [0, 1]


def test_doji():
    df = pd.DataFrame(
        {
            "open": [10, 10],
            "high": [11, 11],
            "low": [9, 9],
            "close": [10.5, 10.05],
        }
    )
    assert doji(df).tolist() == [0, 1]
