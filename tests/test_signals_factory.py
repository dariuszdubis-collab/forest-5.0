import pandas as pd
import pytest

from forest5.config import BacktestSettings

from forest5.examples.synthetic import generate_ohlc
from forest5.signals.factory import compute_signal, _ema_cross_signal
from forest5.core.indicators import rsi
from forest5.signals.candles import candles_signal
from forest5.signals.combine import apply_rsi_filter, confirm_with_candles


def test_compute_signal_ema_cross_basic():
    df = generate_ohlc(periods=60, start_price=100.0)
    s = BacktestSettings()  # default ema_cross
    sig = compute_signal(df, s)

    assert list(sig.index) == list(df.index)  # nosec B101
    assert set(sig.unique()).issubset({-1, 0, 1})  # nosec B101
    assert (sig != 0).any()  # nosec B101


@pytest.mark.parametrize("alias", ["ema_rsi", "ema-cross+rsi"])
def test_compute_signal_does_not_mutate_settings(alias):
    df = generate_ohlc(periods=60, start_price=100.0)
    s = BacktestSettings()
    s.strategy.name = alias
    s.strategy.use_rsi = False

    sig = compute_signal(df, s)

    assert s.strategy.name == alias  # nosec B101
    assert s.strategy.use_rsi is False  # nosec B101
    assert list(sig.index) == list(df.index)  # nosec B101
    assert set(sig.unique()).issubset({-1, 0, 1})  # nosec B101
    assert (sig != 0).any()  # nosec B101


def test_compute_signal_with_rsi_filter_matches_manual():
    df = generate_ohlc(periods=60, start_price=100.0)
    s = BacktestSettings()
    s.strategy.use_rsi = True

    sig = compute_signal(df, s)

    base = _ema_cross_signal(df["close"], s.strategy.fast, s.strategy.slow)
    rsi_series = rsi(df["close"], s.strategy.rsi_period)
    candles = candles_signal(df)
    expected = confirm_with_candles(
        apply_rsi_filter(
            base,
            rsi_series,
            s.strategy.rsi_overbought,
            s.strategy.rsi_oversold,
        ),
        candles,
    )
    assert sig.equals(expected)


def test_compute_signal_macd_cross_basic():
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    prices = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2]
    df = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices},
        index=idx,
    )

    s = BacktestSettings()
    s.strategy.name = "macd_cross"
    s.strategy.fast = 3
    s.strategy.slow = 6
    s.strategy.signal = 3

    sig = compute_signal(df, s)

    assert list(sig.index) == list(df.index)  # nosec B101
    assert set(sig.unique()).issubset({-1, 0, 1})  # nosec B101
    assert 1 in sig.values  # nosec B101
    assert -1 in sig.values  # nosec B101


def test_compute_signal_unknown_strategy_raises():
    df = generate_ohlc(periods=20, start_price=100.0)
    s = BacktestSettings()
    s.strategy.name = "foobar"

    with pytest.raises(ValueError, match="Unknown strategy"):
        compute_signal(df, s)
