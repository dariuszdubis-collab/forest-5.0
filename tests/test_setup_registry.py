from forest5.signals.setups import SetupRegistry, TriggeredSignal
from forest5.signals.contract import TechnicalSignal


def test_setup_registry_triggers_and_clears():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    setup_id = reg.arm("tf", 0, sig)
    # Trigger on next bar
    res = reg.check(1, 11.0)
    assert isinstance(res, TriggeredSignal)
    assert res.setup_id == setup_id
    # After triggering it should be removed
    assert reg.check(1, 11.0) is None


def test_setup_registry_expires_without_trigger():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="SELL", entry=5.0)
    reg.arm("tf", 0, sig)
    # Next bar without breakout, no trigger yet
    assert reg.check(1, 5.5) is None
    # Subsequent bar expires the setup
    assert reg.check(2, 4.0) is None


def test_setup_registry_gap_fill_triggers():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    reg.arm("tf", 0, sig)
    res = reg.check(1, 11.0)
    assert isinstance(res, TriggeredSignal)


def test_setup_registry_blocked_by_time_removes():
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    reg.arm("tf", 0, sig)
    _ = reg.check(1, 11.0)
    assert reg.check(1, 11.0) is None
