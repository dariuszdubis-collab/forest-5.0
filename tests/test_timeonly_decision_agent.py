from datetime import datetime

import pytest

from forest5.decision import DecisionAgent, DecisionConfig


class FakeTimeModel:
    def __init__(self, decision: str) -> None:
        self.decision = decision

    def decide(self, ts, value) -> str:  # pragma: no cover - simple fake
        return self.decision


@pytest.mark.timeonly
@pytest.mark.parametrize(
    ("time_sig", "tech_sig", "expected"),
    [
        ("WAIT", 1, "WAIT"),
        ("BUY", 1, "BUY"),
        ("SELL", -1, "SELL"),
        ("BUY", -1, "WAIT"),
    ],
)
def test_timeonly_decision_agent(time_sig: str, tech_sig: int, expected: str) -> None:
    ts = datetime(2024, 1, 1)
    fake = FakeTimeModel(time_sig)
    agent = DecisionAgent(config=DecisionConfig(min_confluence=2, time_model=fake))
    decision, *_ = agent.decide(ts, tech_signal=tech_sig, value=100.0, symbol="EURUSD")
    assert decision == expected
