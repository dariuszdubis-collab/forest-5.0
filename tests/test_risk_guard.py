import structlog

from forest5.live.risk_guard import RiskGuard, should_halt_for_drawdown
from forest5.utils.log import setup_logger


def _init_logger() -> None:
    structlog.reset_defaults()
    setup_logger()


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


def test_risk_guard_triggers(capfd) -> None:
    _init_logger()
    rg = RiskGuard()
    assert rg.should_halt_for_drawdown(100.0, 75.0, 0.20)
    out, err = capfd.readouterr()
    assert "risk_guard_active" in (out + err)


def test_risk_guard_clears(capfd) -> None:
    _init_logger()
    rg = RiskGuard()
    assert rg.should_halt_for_drawdown(100.0, 75.0, 0.20)
    capfd.readouterr()
    assert not rg.should_halt_for_drawdown(100.0, 95.0, 0.20)
    out, err = capfd.readouterr()
    assert "risk_guard_cleared" in (out + err)
