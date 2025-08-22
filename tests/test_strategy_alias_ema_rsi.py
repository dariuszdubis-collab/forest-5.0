from forest5.backtest.engine import _generate_signal
from forest5.config import BacktestSettings
from forest5.examples.synthetic import generate_ohlc


def test_strategy_alias_does_not_mutate_settings() -> None:
    df = generate_ohlc(periods=40, start_price=100.0)
    settings = BacktestSettings()
    settings.strategy.name = "ema_rsi"  # alias for ema_cross with RSI
    settings.strategy.use_rsi = False

    sig = _generate_signal(df, settings, price_col="close")

    assert settings.strategy.name == "ema_rsi"
    assert settings.strategy.use_rsi is False
    assert list(sig.index) == list(df.index)  # nosec B101
    assert set(sig.unique()).issubset({-1, 0, 1})  # nosec B101
