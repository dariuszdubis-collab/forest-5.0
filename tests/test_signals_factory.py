import pandas as pd
import pytest
from types import SimpleNamespace

from forest5.config import BacktestSettings

from forest5.examples.synthetic import generate_ohlc
from forest5.signals.compat import compute_signal_compat
from forest5.signals.ema import ema_cross_signal
from forest5.core.indicators import rsi
from forest5.signals.candles import candles_signal
from forest5.signals.combine import apply_rsi_filter, confirm_with_candles
from forest5.signals.factory import compute_signal
from forest5.signals.contract import TechnicalSignal


def test_compute_signal_ema_cross_basic():
    df = generate_ohlc(periods=60, start_price=100.0)
    s = BacktestSettings()  # default ema_cross
    sig = compute_signal_compat(df, s)

    assert list(sig.index) == list(df.index)  # nosec B101
    assert set(sig.unique()).issubset({-1, 0, 1})  # nosec B101
    assert (sig != 0).any()  # nosec B101


@pytest.mark.parametrize("alias", ["ema_rsi", "ema-cross+rsi"])
def test_compute_signal_does_not_mutate_settings(alias):
    df = generate_ohlc(periods=60, start_price=100.0)
    s = BacktestSettings()
    s.strategy.name = alias
    s.strategy.use_rsi = False

    sig = compute_signal_compat(df, s)

    assert s.strategy.name == alias  # nosec B101
    assert s.strategy.use_rsi is False  # nosec B101
    assert list(sig.index) == list(df.index)  # nosec B101
    assert set(sig.unique()).issubset({-1, 0, 1})  # nosec B101
    assert (sig != 0).any()  # nosec B101


def test_compute_signal_with_rsi_filter_matches_manual():
    df = generate_ohlc(periods=60, start_price=100.0)
    s = BacktestSettings()
    s.strategy.use_rsi = True

    sig = compute_signal_compat(df, s)

    base = ema_cross_signal(df["close"], s.strategy.fast, s.strategy.slow)
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

    sig = compute_signal_compat(df, s)

    assert list(sig.index) == list(df.index)  # nosec B101
    assert set(sig.unique()).issubset({-1, 0, 1})  # nosec B101
    assert 1 in sig.values  # nosec B101
    assert -1 in sig.values  # nosec B101


def test_compute_signal_unknown_strategy_raises():
    df = generate_ohlc(periods=20, start_price=100.0)
    s = BacktestSettings()
    s.strategy.name = "foobar"

    with pytest.raises(ValueError, match="Unknown strategy"):
        compute_signal_compat(df, s)


def _h1_settings(**kwargs):
    strat = SimpleNamespace(name="h1_ema_rsi_atr", params=None, compat_int=False)
    for k, v in kwargs.items():
        setattr(strat, k, v)
    return SimpleNamespace(strategy=strat)


def test_compute_signal_h1_returns_contract():
    df = generate_ohlc(periods=100, start_price=100.0)
    s = _h1_settings()
    sig = compute_signal(df, s)
    assert isinstance(sig, TechnicalSignal)


def test_compute_signal_h1_compat_int():
    df = generate_ohlc(periods=100, start_price=100.0)
    s = _h1_settings()
    sig = compute_signal(df, s, compat_int=True)
    assert isinstance(sig, int) and sig in (-1, 0, 1)
    series = compute_signal_compat(df, s)
    assert list(series.index) == list(df.index)  # nosec B101
    assert set(series.unique()).issubset({-1, 0, 1})  # nosec B101


def test_compute_signal_compat_handles_contract():
    df = generate_ohlc(periods=100, start_price=100.0)
    s = _h1_settings()
    series = compute_signal_compat(df, s)
    assert list(series.index) == list(df.index)  # nosec B101
    assert set(series.unique()).issubset({-1, 0, 1})  # nosec B101


def test_compute_signal_compat_accepts_mapping(monkeypatch):
    df = generate_ohlc(periods=10, start_price=100.0)
    s = BacktestSettings()

    def fake_compute(df, settings, price_col="close"):
        return {"action": "BUY"}

    monkeypatch.setattr("forest5.signals.compat.compute_signal", fake_compute)
    series = compute_signal_compat(df, s)
    assert list(series.index) == list(df.index)  # nosec B101
    assert series.iloc[-1] == 1  # nosec B101


def test_compute_signal_passes_contract(monkeypatch):
    df = generate_ohlc(periods=10, start_price=100.0)
    s = BacktestSettings()
    s.strategy.name = "h1_ema_rsi_atr"
    fake = TechnicalSignal(action="BUY", entry=1.0, sl=0.5, tp=1.5)

    def _fake(df, params, ctx=None):
        return fake

    monkeypatch.setattr("forest5.signals.factory.compute_primary_signal_h1", _fake)
    res = compute_signal(df, s)
    assert res is fake
