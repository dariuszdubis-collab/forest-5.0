from forest5.live.risk_guard import should_halt_for_drawdown


def test_should_halt_for_drawdown_basic() -> None:
    assert not should_halt_for_drawdown(100.0, 95.0, 0.10)
    assert should_halt_for_drawdown(100.0, 75.0, 0.20)
    assert not should_halt_for_drawdown(0.0, 80.0, 0.20)


def test_should_halt_for_drawdown_edge_cases() -> None:
    # hitting the threshold exactly triggers halt
    assert should_halt_for_drawdown(100.0, 80.0, 0.20)
    # non-positive max_dd never halts
    assert not should_halt_for_drawdown(100.0, 50.0, 0.0)
    assert not should_halt_for_drawdown(100.0, 50.0, -0.1)
    # profit scenario does not halt
    assert not should_halt_for_drawdown(100.0, 120.0, 0.10)
    # negative current equity counts toward drawdown
    assert should_halt_for_drawdown(100.0, -10.0, 0.5)
