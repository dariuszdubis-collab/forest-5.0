from datetime import datetime

from forest5.decision import DecisionAgent, DecisionConfig
from forest5.numerology import NumerologyRules


def test_decision_agent_wait_on_blocked_weekday() -> None:
    rules = NumerologyRules(enabled=True, blocked_weekdays=[0])
    config = DecisionConfig(numerology=rules)
    agent = DecisionAgent(config=config)

    ts = datetime(2024, 1, 1)  # Monday
    decision = agent.decide(ts, tech_signal=1, symbol="EURUSD")

    assert decision == "WAIT"
