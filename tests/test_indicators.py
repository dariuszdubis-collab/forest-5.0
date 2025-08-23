"""Unit tests for indicator helpers."""

import pandas as pd
from forest5.core.indicators import atr, atr_at_close, ema, rsi_cross

def test_ema_simple():  # fmt: skip
    s = pd.Series([1, 2, 3, 4, 5])
    e = ema(s, 2)
    assert e.isna().sum() == 0
    assert abs(float(e.iloc[-1]) - 4.0) < 1.0  # „rozsądnie blisko”


def test_atr_positive():
    high = pd.Series([10, 11, 12, 13])
    low = pd.Series([9, 9.5, 10, 11])
    close = pd.Series([9.5, 10.5, 11, 12])
    a = atr(high, low, close, 3)
    assert (a >= 0).all()


def test_atr_at_close_masks_non_closes():
    high = pd.Series([10, 11, 12, 13])
    low = pd.Series([9, 9.5, 10, 11])
    close = pd.Series([9.5, float("nan"), float("nan"), 12])
    a = atr_at_close(high, low, close, 3)
    ref = atr(high, low, close.ffill(), 3)
    assert a.isna().sum() == 2
    assert a.iloc[0] == ref.iloc[0]
    assert a.iloc[-1] == ref.iloc[-1]


def test_rsi_cross_events():
    rsi = pd.Series([45, 49, 51, 52, 48, 49, 51])
    signals = rsi_cross(rsi)
    assert list(signals) == [0, 0, 1, 0, -1, 0, 1]
