from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from forest5.decision import DecisionAgent, DecisionConfig


class DummyTimeModel:
    def __init__(self, decision: Literal["BUY", "SELL", "WAIT"]) -> None:
        self.decision = decision

    def decide(
        self, ts, value: float
    ) -> Literal["BUY", "SELL", "WAIT"]:  # pragma: no cover - trivial
        return self.decision


def test_wait_short_circuit() -> None:
    tm = DummyTimeModel("WAIT")
    agent = DecisionAgent(config=DecisionConfig(time_model=tm, min_confluence=2))
    ts = datetime(2024, 1, 1)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {"tech": 1},
        "timeonly_wait",
    )


def test_min_confluence_requires_both_votes() -> None:
    tm = DummyTimeModel("BUY")
    agent = DecisionAgent(config=DecisionConfig(time_model=tm, min_confluence=2))
    ts = datetime(2024, 1, 1)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "time": 1},
        "buy_majority",
    )
    decision, votes, reason = agent.decide(ts, tech_signal=0, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {"tech": 0, "time": 1},
        "no_consensus",
    )


def test_min_confluence_sell_and_conflict_without_ai() -> None:
    ts = datetime(2024, 1, 1)
    tm_sell = DummyTimeModel("SELL")
    agent_sell = DecisionAgent(config=DecisionConfig(time_model=tm_sell, min_confluence=2))
    decision, votes, reason = agent_sell.decide(ts, tech_signal=-1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "SELL",
        {"tech": -1, "time": -1},
        "sell_majority",
    )
    tm_buy = DummyTimeModel("BUY")
    agent_conflict = DecisionAgent(config=DecisionConfig(time_model=tm_buy, min_confluence=2))
    decision, votes, reason = agent_conflict.decide(ts, tech_signal=-1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {"tech": -1, "time": 1},
        "no_consensus",
    )


@dataclass
class DummySentiment:
    score: float


class DummyAI:
    def __init__(self, score: float) -> None:
        self._score = score

    def analyse(self, context: str, symbol: str) -> DummySentiment:  # pragma: no cover - trivial
        return DummySentiment(self._score)


def test_min_confluence_with_ai() -> None:
    ts = datetime(2024, 1, 1)
    agent = DecisionAgent(config=DecisionConfig(use_ai=True, min_confluence=2))

    agent.ai = DummyAI(1.0)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "ai": 1},
        "buy_majority",
    )

    agent.ai = DummyAI(-1.0)
    decision, votes, reason = agent.decide(ts, tech_signal=-1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "SELL",
        {"tech": -1, "ai": -1},
        "sell_majority",
    )

    agent.ai = DummyAI(-1.0)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {"tech": 1, "ai": -1},
        "no_consensus",
    )
