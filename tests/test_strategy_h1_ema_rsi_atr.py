from forest5.config.strategy import H1EmaRsiAtrSettings


def test_h1_ema_rsi_atr_settings_defaults():
    s = H1EmaRsiAtrSettings()
    assert s.name == "h1_ema_rsi_atr"  # nosec B101
    assert s.params.ema_fast == 21  # nosec B101
    assert s.params.ema_slow == 55  # nosec B101
