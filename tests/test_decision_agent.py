from datetime import datetime

from forest5.decision import DecisionAgent, DecisionConfig


class FakeTimeModel:
    def __init__(self, decision: str = "HOLD", weight: float = 1.0) -> None:
        self.decision = decision
        self.weight = weight

    def decide(self, ts):  # pragma: no cover - simple fake
        return {"decision": self.decision, "weight": self.weight}


def test_decision_agent_waits_when_time_model_waits() -> None:
    time_model = FakeTimeModel("WAIT")
    agent = DecisionAgent(config=DecisionConfig(time_model=time_model))

    ts = datetime(2024, 1, 1)  # 00:00
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=0.0, symbol="EURUSD")
    assert decision == "WAIT"
    assert votes == {"tech": 1, "time": 0, "ai": 0}
    assert reason == "time_wait"

def test_decision_agent_majority_and_tie() -> None:
    time_model = FakeTimeModel()
    agent = DecisionAgent(config=DecisionConfig(time_model=time_model))
    ts = datetime(2024, 1, 1)

    time_model.decision = "BUY"
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "time": 1, "ai": 0},
        "buy_majority",
    )

    time_model.decision = "SELL"
    decision, votes, reason = agent.decide(ts, tech_signal=-1, value=-2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "SELL",
        {"tech": -1, "time": -1, "ai": 0},
        "sell_majority",
    )

    time_model.decision = "SELL"
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=-2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {"tech": 1, "time": -1, "ai": 0},
        "no_consensus",
    )


def test_decision_agent_time_hold_is_neutral() -> None:
    time_model = FakeTimeModel("HOLD")
    agent = DecisionAgent(config=DecisionConfig(time_model=time_model))
    ts = datetime(2024, 1, 1)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=0.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "time": 0, "ai": 0},
        "buy_majority",
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

    time_model = FakeTimeModel("BUY")
    agent2 = DecisionAgent(config=DecisionConfig(time_model=time_model, min_confluence=2))
    decision, votes, reason = agent2.decide(ts, tech_signal=1, value=2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "time": 1, "ai": 0},
        "buy_majority",
    )
