from datetime import datetime

from forest5.decision import DecisionAgent, DecisionConfig
from forest5.time_only import TimeOnlyModel


def test_decision_agent_waits_when_time_model_waits() -> None:
    time_model = TimeOnlyModel({0: (-1.0, 1.0)}, q_low=-1.0, q_high=1.0)
    agent = DecisionAgent(config=DecisionConfig(time_model=time_model))

    ts = datetime(2024, 1, 1)  # 00:00
    assert (
        agent.decide(ts, tech_signal=1, value=0.0, symbol="EURUSD") == "WAIT"
    )
