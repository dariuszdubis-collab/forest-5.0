"""Unit tests for array-based indicators."""

import numpy as np
from forest5.core.indicators import ema, atr, rsi, atr_offset


def test_ema_simple():
    arr = np.array([1, 2, 3, 4, 5], dtype=float)
    e = ema(arr, 2)
    assert e.shape == arr.shape
    assert not np.isnan(e).any()
    assert abs(float(e[-1]) - 4.0) < 1.0


def test_atr_positive():
    high = np.array([10, 11, 12, 13], dtype=float)
    low = np.array([9, 9.5, 10, 11], dtype=float)
    close = np.array([9.5, 10.5, 11, 12], dtype=float)
    a = atr(high, low, close, 3)
    assert a.shape == high.shape
    assert np.all(a >= 0)


def test_rsi_bounds():
    close = np.array([1, 2, 1.5, 1.8, 1.2, 1.4, 1.3, 1.6, 1.1, 1.5])
    r = rsi(close, 3)
    assert r.shape == close.shape
    assert np.nanmax(r) <= 100
    assert np.nanmin(r) >= 0


def test_atr_offset():
    assert atr_offset(100.0, 2.0, 1.5) == 103.0
