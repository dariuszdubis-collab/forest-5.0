from datetime import datetime

from forest5.decision import DecisionAgent, DecisionConfig
from forest5.time_only import TimeOnlyModel


def test_core_fusion_time_wait() -> None:
    tm = TimeOnlyModel({0: (-1.0, 1.0)}, q_low=-1.0, q_high=1.0)
    agent = DecisionAgent(config=DecisionConfig(time_model=tm, use_core_fusion=True))
    ts = datetime(2024, 1, 1)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=0.0, symbol="EURUSD")
    assert (decision, votes, reason) == ("WAIT", {}, "timeonly_wait")


def test_core_fusion_min_confluence() -> None:
    agent = DecisionAgent(config=DecisionConfig(min_confluence=1.4, use_core_fusion=True))
    ts = datetime(2024, 1, 1)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=0.0, symbol="EURUSD")
    # With only tech vote=1 and min_confluence=1.4, expect WAIT
    assert (decision, votes, reason) == ("WAIT", {}, "below_min_confluence")
