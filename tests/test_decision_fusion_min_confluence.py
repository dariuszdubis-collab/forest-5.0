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
    agent = DecisionAgent(config=DecisionConfig(time_model=tm, min_confluence=1.4))
    ts = datetime(2024, 1, 1)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {},
        "timeonly_wait",
    )


def test_min_confluence_requires_both_votes() -> None:
    tm = DummyTimeModel("BUY")
    agent = DecisionAgent(config=DecisionConfig(time_model=tm, min_confluence=1.4))
    ts = datetime(2024, 1, 1)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "time": 1},
        "ok",
    )
    agent2 = DecisionAgent(config=DecisionConfig(min_confluence=1.4))
    decision, votes, reason = agent2.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {},
        "below_min_confluence",
    )


def test_min_confluence_sell_and_conflict_without_ai() -> None:
    ts = datetime(2024, 1, 1)
    tm_sell = DummyTimeModel("SELL")
    cfg_sell = DecisionConfig(time_model=tm_sell, min_confluence=1.0)
    cfg_sell.weights.time = 0.5
    agent_sell = DecisionAgent(config=cfg_sell)
    decision, votes, reason = agent_sell.decide(ts, tech_signal=-1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "SELL",
        {"tech": -1, "time": -1},
        "ok",
    )
    tm_buy = DummyTimeModel("BUY")
    cfg_conflict = DecisionConfig(time_model=tm_buy, min_confluence=1.0)
    cfg_conflict.weights.time = 0.5
    agent_conflict = DecisionAgent(config=cfg_conflict)
    decision, votes, reason = agent_conflict.decide(ts, tech_signal=-1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {},
        "tie",
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
    agent = DecisionAgent(config=DecisionConfig(use_ai=True, min_confluence=1.0))

    agent.ai = DummyAI(1.0)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "ai": 1},
        "ok",
    )

    agent.ai = DummyAI(-1.0)
    decision, votes, reason = agent.decide(ts, tech_signal=-1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "SELL",
        {"tech": -1, "ai": -1},
        "ok",
    )

    agent.ai = DummyAI(-1.0)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {},
        "tie",
    )
