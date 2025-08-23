import numpy as np
import pandas as pd
from forest5.core.indicators import atr, atr_offset, ema, rsi


def test_ema_array_series_equivalence():
    data = [1, 2, 3, 4, 5]
    expected = np.array(
        [1.0, 1.6666666666666665, 2.5555555555555554, 3.518518518518518, 4.506172839506172]
    )

    s = pd.Series(data)
    e_series = ema(s, 2)
    assert isinstance(e_series, pd.Series)
    np.testing.assert_allclose(e_series.to_numpy(), expected)

    e_array = ema(np.array(data), 2)
    assert isinstance(e_array, np.ndarray)
    np.testing.assert_allclose(e_array, expected)


def test_atr_array_series_equivalence():
    high = [10, 11, 12, 13]
    low = [9, 9.5, 10, 11]
    close = [9.5, 10.5, 11, 12]
    expected = np.array(
        [1.0, 1.1666666666666665, 1.4444444444444442, 1.6296296296296293]
    )

    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    a_series = atr(high_s, low_s, close_s, 3)
    assert isinstance(a_series, pd.Series)
    np.testing.assert_allclose(a_series.to_numpy(), expected)

    a_array = atr(np.array(high), np.array(low), np.array(close), 3)
    assert isinstance(a_array, np.ndarray)
    np.testing.assert_allclose(a_array, expected)


def test_rsi_array_series_equivalence():
    close = [1, 2, 3, 4, 5, 6, 5, 4, 3]
    expected = np.array(
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 80.0, 64.0, 51.20000000000001]
    )

    close_s = pd.Series(close)
    r_series = rsi(close_s, 5)
    assert isinstance(r_series, pd.Series)
    np.testing.assert_allclose(r_series.to_numpy(), expected, equal_nan=True)

    r_array = rsi(np.array(close), 5)
    assert isinstance(r_array, np.ndarray)
    np.testing.assert_allclose(r_array, expected, equal_nan=True)


def test_atr_offset():
    assert atr_offset(100.0, 2.0, 5.0) == 110.0
    assert atr_offset(100.0, -2.0, 5.0) == 90.0
