from forest5.signals.setups import SetupRegistry
from forest5.signals.contract import TechnicalSignal


def test_setup_registry_triggers_and_clears():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    reg.arm("tf", 0, sig)
    # Trigger on next bar
    res = reg.check(index=1, price=11.0)
    assert isinstance(res, TechnicalSignal)
    # After triggering it should be removed
    assert reg.check(index=1, price=11.0) is None


def test_setup_registry_expires_without_trigger():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="SELL", entry=5.0)
    reg.arm("tf", 0, sig)
    # Next bar without breakout expires the setup
    assert reg.check(index=1, price=5.2) is None
    # Subsequent checks remain empty
    assert reg.check(index=2, price=4.0) is None


def test_setup_registry_gap_fill_triggers():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    reg.arm("tf", 0, sig)
    res = reg.check(index=1, price=11.0)
    assert isinstance(res, TechnicalSignal)


def test_setup_registry_blocked_by_time_removes():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    reg.arm("tf", 0, sig)
    _ = reg.check(index=1, price=11.0)
    assert reg.check(index=1, price=11.0) is None
