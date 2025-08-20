import logging

from forest5.live.risk_guard import should_halt_for_drawdown
from forest5.utils.log import log


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


def _update_risk_guard(risk_halt: bool, cur_eq: float) -> bool:
    start_eq = 100.0
    max_dd = 0.20
    if should_halt_for_drawdown(start_eq, cur_eq, max_dd):
        dd = (start_eq - cur_eq) / start_eq
        if not risk_halt:
            log.info("risk_guard_active", drawdown_pct=dd * 100)
        return True
    if risk_halt:
        dd = (start_eq - cur_eq) / start_eq
        log.info("risk_guard_cleared", drawdown_pct=dd * 100)
    return False


def test_risk_guard_triggers(capfd) -> None:
    risk_halt = _update_risk_guard(False, 75.0)
    assert risk_halt
    out, _ = capfd.readouterr()
    assert "risk_guard_active" in out


def test_risk_guard_clears(capfd) -> None:
    risk_halt = _update_risk_guard(True, 95.0)
    assert not risk_halt
    out, _ = capfd.readouterr()
    assert "risk_guard_cleared" in out
