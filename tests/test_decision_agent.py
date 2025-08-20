from datetime import datetime

from forest5.decision import DecisionAgent, DecisionConfig
from forest5.time_only import TimeOnlyModel


def test_decision_agent_waits_when_time_model_waits() -> None:
    time_model = TimeOnlyModel({0: (-1.0, 1.0)}, q_low=-1.0, q_high=1.0)
    agent = DecisionAgent(config=DecisionConfig(time_model=time_model))

    ts = datetime(2024, 1, 1)  # 00:00
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=0.0, symbol="EURUSD")
    assert decision == "WAIT"
    assert votes == {"tech": 1, "time": 0, "ai": 0}
    assert reason == "time_wait"


def test_decision_agent_majority_and_tie() -> None:
    time_model = TimeOnlyModel({0: (-1.0, 1.0)}, q_low=-1.0, q_high=1.0)
    agent = DecisionAgent(config=DecisionConfig(time_model=time_model))
    ts = datetime(2024, 1, 1)

    decision, votes, reason = agent.decide(ts, tech_signal=1, value=2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "time": 1, "ai": 0},
        "buy_majority",
    )
    decision, votes, reason = agent.decide(ts, tech_signal=-1, value=-2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "SELL",
        {"tech": -1, "time": -1, "ai": 0},
        "sell_majority",
    )
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=-2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {"tech": 1, "time": -1, "ai": 0},
        "no_consensus",
    )


def test_decision_agent_respects_confluence_threshold() -> None:
    ts = datetime(2024, 1, 1)
    agent = DecisionAgent(config=DecisionConfig(min_confluence=2))

    decision, votes, reason = agent.decide(ts, tech_signal=1, value=0.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {"tech": 1, "time": 0, "ai": 0},
        "no_consensus",
    )

    time_model = TimeOnlyModel({0: (-1.0, 1.0)}, q_low=-1.0, q_high=1.0)
    agent2 = DecisionAgent(config=DecisionConfig(time_model=time_model, min_confluence=2))
    decision, votes, reason = agent2.decide(ts, tech_signal=1, value=2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "time": 1, "ai": 0},
        "buy_majority",
    )
