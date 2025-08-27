from pathlib import Path

from forest5.live.live_runner import run_live
from forest5.live.settings import (
    LiveSettings,
    BrokerSettings,
    DecisionSettings,
    AISettings,
    TimeSettings,
    RiskSettings,
)


def test_missing_ai_context_disables(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    settings = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=str(tmp_path), symbol="EURUSD", volume=0.01),
        decision=DecisionSettings(min_confluence=0.0),
        ai=AISettings(
            enabled=True,
            model="gpt-4o-mini",
            max_tokens=32,
            context_file=str(tmp_path / "nope.txt"),
            require_context=False,
        ),
        time=TimeSettings(),
        risk=RiskSettings(max_drawdown=0.5),
    )

    run_live(settings, max_steps=0, timeout=0)
    out = capsys.readouterr().out
    assert "ai_context_missing_warn" in out
