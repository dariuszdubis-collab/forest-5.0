from forest5.signals.setups import SetupRegistry
from forest5.signals.contract import TechnicalSignal


def test_setup_registry_triggers_and_clears():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    reg.arm("tf", 0, sig)
    # Trigger on next bar
    res = reg.check("tf", 1, high=11.0, low=9.0)
    assert res is sig
    # After triggering it should be removed
    assert reg.check("tf", 1, high=11.0, low=9.0) is None


def test_setup_registry_expires_without_trigger():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="SELL", entry=5.0)
    reg.arm("tf", 0, sig)
    # Next bar without breakout expires the setup
    assert reg.check("tf", 1, high=5.5, low=5.1) is None
    # Subsequent checks remain empty
    assert reg.check("tf", 2, high=4.0, low=3.0) is None


def test_setup_registry_gap_fill_triggers():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    reg.arm("tf", 0, sig)
    res = reg.check("tf", 1, high=11.0, low=11.0)
    assert res is sig


def test_setup_registry_blocked_by_time_removes():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    reg.arm("tf", 0, sig)
    _ = reg.check("tf", 1, high=11.0, low=9.0)
    assert reg.check("tf", 1, high=11.0, low=9.0) is None
