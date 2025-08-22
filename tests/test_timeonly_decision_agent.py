from datetime import datetime

import pytest

from forest5.decision import DecisionAgent, DecisionConfig


class FakeTimeModel:
    def __init__(self, decision: str, weight: float = 0.5) -> None:
        self.decision = decision
        self.weight = weight

    def decide(self, ts):  # pragma: no cover - simple fake
        return {"decision": self.decision, "weight": self.weight}


@pytest.mark.timeonly
@pytest.mark.parametrize(
    ("time_sig", "tech_sig", "expected"),
    [
        ("WAIT", 1, "WAIT"),
        ("BUY", 1, "BUY"),
        ("SELL", -1, "SELL"),
        ("BUY", -1, "SELL"),  # tech signal stronger
        ("HOLD", 1, "BUY"),
    ],
)
def test_timeonly_decision_agent(time_sig: str, tech_sig: int, expected: str) -> None:
    ts = datetime(2024, 1, 1)
    fake = FakeTimeModel(time_sig)
    agent = DecisionAgent(config=DecisionConfig(min_confluence=1.0, time_model=fake))
    res = agent.decide(ts, tech_signal=tech_sig, value=100.0, symbol="EURUSD")
    assert res.decision == expected
    exp_weight = 0.0 if expected == "WAIT" else (
        0.5 if time_sig == expected and time_sig != "HOLD" else 1.0
    )
    assert res.weight == pytest.approx(exp_weight)
