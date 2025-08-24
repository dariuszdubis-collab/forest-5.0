from pathlib import Path
from forest5.config.loader import load_live_settings


def test_live_example_yaml_parses_and_has_fields():
    s = load_live_settings(Path("config/live.example.yaml"))
    assert hasattr(s, "broker")  # nosec B101
    assert hasattr(s, "ai")  # nosec B101
    assert hasattr(s, "time")  # nosec B101
    assert hasattr(s, "decision")  # nosec B101
    assert getattr(s.decision, "min_confluence", None) is not None  # nosec B101
    assert s.decision.min_confluence >= 0  # nosec B101
    assert hasattr(s.ai, "context_file")  # nosec B101
    tm = s.time.model if hasattr(s.time, "model") else None
    assert tm is not None  # nosec B101
    assert hasattr(tm, "enabled")  # nosec B101
    assert hasattr(tm, "path")  # nosec B101
    assert hasattr(tm, "q_low")  # nosec B101
    assert hasattr(tm, "q_high")  # nosec B101
    ps = s.time.primary_signal if hasattr(s.time, "primary_signal") else None
    assert ps is not None  # nosec B101
    assert hasattr(ps, "strategy")  # nosec B101
    assert getattr(ps.strategy, "compat_int", None) is not None  # nosec B101
    pats = ps.patterns
    assert hasattr(pats, "engulf") and pats.engulf.enabled is True  # nosec B101
    assert hasattr(pats, "pinbar") and pats.pinbar.enabled is True  # nosec B101
    assert hasattr(pats, "star") and pats.star.enabled is True  # nosec B101
    assert hasattr(s, "risk")  # nosec B101
    assert getattr(s.risk, "on_drawdown", None) is not None  # nosec B101
    assert s.risk.on_drawdown.action == "halt"  # nosec B101
