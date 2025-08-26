import pytest

from datetime import datetime, timedelta

from forest5.signals.setups import SetupRegistry, TriggeredSignal
from forest5.signals.contract import TechnicalSignal


def test_setup_registry_triggers_and_clears() -> None:
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    t0 = datetime(2024, 1, 1)
    setup_id = reg.arm("tf", 0, sig, bar_time=t0)
    res = reg.check(index=1, price=11.0, now=t0)
    assert isinstance(res, TriggeredSignal)
    assert res.setup_id == setup_id
    assert reg.check(index=1, price=11.0, now=t0) is None


def test_setup_registry_expires_without_trigger() -> None:
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="SELL", entry=5.0)
    t0 = datetime(2024, 1, 1)
    reg.arm("tf", 0, sig, bar_time=t0)
    assert reg.check(index=1, price=5.2, now=t0) is None
    assert reg.check(index=2, price=4.0, now=t0) is None


def test_setup_registry_gap_fill_triggers() -> None:
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    t0 = datetime(2024, 1, 1)
    reg.arm("tf", 0, sig, bar_time=t0)
    res = reg.check(index=1, price=11.0, now=t0)
    assert isinstance(res, TriggeredSignal)


def test_setup_registry_blocked_by_time_removes() -> None:
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    t0 = datetime(2024, 1, 1)
    reg.arm("tf", 0, sig, bar_time=t0)
    _ = reg.check(index=1, price=11.0, now=t0)
    assert reg.check(index=1, price=11.0, now=t0) is None


def test_setup_registry_returns_trigger_details() -> None:
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    t0 = datetime(2024, 1, 1)
    reg.arm("tf", 0, sig, bar_time=t0)
    res = reg.check(index=1, price=10.5, now=t0)
    assert isinstance(res, TriggeredSignal)
    assert res.fill_price == 10.5
    assert res.slippage == pytest.approx(0.5)


def test_setup_registry_high_low_trigger_and_expire() -> None:
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    t0 = datetime(2024, 1, 1)
    reg.arm("tf", 0, sig, bar_time=t0)
    res = reg.check(index=1, high=10.5, low=9.0, now=t0)
    assert isinstance(res, TriggeredSignal)
    assert reg.check(index=1, price=11.0, now=t0) is None


def test_setup_registry_low_trigger_sell() -> None:
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="SELL", entry=5.0)
    t0 = datetime(2024, 1, 1)
    reg.arm("tf", 0, sig, bar_time=t0)
    res = reg.check(index=1, low=4.8, now=t0)
    assert isinstance(res, TriggeredSignal)


def test_setup_registry_high_low_expire_without_trigger() -> None:
    reg = SetupRegistry(ttl_bars=1)
    sig = TechnicalSignal(action="BUY", entry=10.0)
    t0 = datetime(2024, 1, 1)
    reg.arm("tf", 0, sig, bar_time=t0)
    assert reg.check(index=1, high=9.5, low=9.0, now=t0) is None
    assert reg.check(index=2, price=11.0, now=t0) is None


def test_setup_registry_ttl_minutes_expires() -> None:
    reg = SetupRegistry(ttl_bars=10)
    sig = TechnicalSignal(action="BUY", entry=10.0, ttl_minutes=1)
    t0 = datetime(2024, 1, 1, 0, 0)
    reg.arm("tf", 0, sig, bar_time=t0)
    t1 = t0 + timedelta(minutes=2)
    assert reg.check(index=1, price=9.0, now=t1) is None
    assert reg.check(index=2, price=11.0, now=t1) is None
