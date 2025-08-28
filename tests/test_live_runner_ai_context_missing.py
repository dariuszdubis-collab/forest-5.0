from pathlib import Path

from forest5.live.settings import (
    LiveSettings,
    BrokerSettings,
    DecisionSettings,
    AISettings,
    TimeSettings,
    RiskSettings,
)
import pytest


def test_missing_ai_context_errors(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    with pytest.raises(Exception, match="ai.context_file missing"):
        LiveSettings(
            broker=BrokerSettings(
                type="paper", bridge_dir=str(tmp_path), symbol="EURUSD", volume=0.01
            ),
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
