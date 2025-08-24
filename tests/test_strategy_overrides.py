from forest5.config.strategy import BaseStrategySettings, H1EmaRsiAtrParams


def test_overrides_propagate_to_params():
    s = BaseStrategySettings(
        name="h1_ema_rsi_atr",
        ema_fast=10,
        pullback_atr=0.7,
        entry_buffer_atr=0.2,
    )
    assert isinstance(s.params, H1EmaRsiAtrParams)
    assert s.params.ema_fast == 10
    assert s.params.pullback_atr == 0.7
    assert s.params.entry_buffer_atr == 0.2


def test_alias_input_propagates():
    s = BaseStrategySettings(name="h1_ema_rsi_atr", **{"pullback_to_ema_fast_atr": 0.8})
    assert s.pullback_atr == 0.8
    assert s.params.pullback_atr == 0.8
