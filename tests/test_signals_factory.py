import pytest

from forest5.config import BacktestSettings
from forest5.examples.synthetic import generate_ohlc
from forest5.signals.factory import compute_signal


def test_compute_signal_ema_cross_basic():
    df = generate_ohlc(periods=60, start_price=100.0)
    s = BacktestSettings()  # domyślnie ema_cross
    sig = compute_signal(df, s)

    # ta sama długość i indeks co dane
    assert list(sig.index) == list(df.index)  # nosec B101
    # dopuszczalne wartości
    assert set(sig.unique()).issubset({-1, 0, 1})  # nosec B101
    # nie wszystkie zera (powinny zdarzyć się przecięcia)
    assert (sig != 0).any()  # nosec B101


@pytest.mark.parametrize("alias", ["ema_rsi", "ema-cross+rsi"])
def test_compute_signal_normalizes_rsi_alias(alias):
    df = generate_ohlc(periods=60, start_price=100.0)
    s = BacktestSettings()
    s.strategy.name = alias
    s.strategy.use_rsi = False

    sig = compute_signal(df, s)

    assert s.strategy.name == "ema_cross"  # nosec B101
    assert s.strategy.use_rsi is True  # nosec B101
    assert list(sig.index) == list(df.index)  # nosec B101
    assert set(sig.unique()).issubset({-1, 0, 1})  # nosec B101
    assert (sig != 0).any()  # nosec B101
