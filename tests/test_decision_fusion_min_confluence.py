from datetime import datetime

from forest5.decision import DecisionAgent, DecisionConfig


class DummyTimeModel:
    def __init__(self, decision: str) -> None:
        self.decision = decision

    def decide(self, ts, value: float) -> str:  # pragma: no cover - trivial
        return self.decision


def test_wait_short_circuit() -> None:
    tm = DummyTimeModel("WAIT")
    agent = DecisionAgent(config=DecisionConfig(time_model=tm, min_confluence=2))
    ts = datetime(2024, 1, 1)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {"tech": 1, "time": 0, "ai": 0},
        "time_wait",
    )


def test_min_confluence_requires_both_votes() -> None:
    tm = DummyTimeModel("BUY")
    agent = DecisionAgent(config=DecisionConfig(time_model=tm, min_confluence=2))
    ts = datetime(2024, 1, 1)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "time": 1, "ai": 0},
        "buy_majority",
    )
    decision, votes, reason = agent.decide(ts, tech_signal=0, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {"tech": 0, "time": 1, "ai": 0},
        "no_consensus",
    )

