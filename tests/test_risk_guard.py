from forest5.live.risk_guard import should_halt_for_drawdown


def test_should_halt_for_drawdown_basic():
    assert not should_halt_for_drawdown(100.0, 95.0, 0.10)
    assert should_halt_for_drawdown(100.0, 75.0, 0.20)
    assert not should_halt_for_drawdown(0.0, 80.0, 0.20)
