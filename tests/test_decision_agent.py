from dataclasses import dataclass
from datetime import datetime

from forest5.decision import DecisionAgent, DecisionConfig
from forest5.time_only import TimeOnlyModel
from forest5.signals.contract import TechnicalSignal


def test_decision_agent_waits_when_time_model_waits() -> None:
    time_model = TimeOnlyModel({0: (-1.0, 1.0)}, q_low=-1.0, q_high=1.0)
    agent = DecisionAgent(config=DecisionConfig(time_model=time_model))

    ts = datetime(2024, 1, 1)  # 00:00
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=0.0, symbol="EURUSD")
    assert decision == "WAIT"
    assert votes == {}
    assert reason == "timeonly_wait"


def test_decision_agent_majority_and_tie() -> None:
    time_model = TimeOnlyModel({0: (-1.0, 1.0)}, q_low=-1.0, q_high=1.0)
    cfg = DecisionConfig(time_model=time_model)
    cfg.weights.time = 0.5
    agent = DecisionAgent(config=cfg)
    ts = datetime(2024, 1, 1)

    decision, votes, reason = agent.decide(ts, tech_signal=1, value=2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "time": 1},
        "ok",
    )
    decision, votes, reason = agent.decide(ts, tech_signal=-1, value=-2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "SELL",
        {"tech": -1, "time": -1},
        "ok",
    )
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=-2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {},
        "tie",
    )


def test_decision_agent_respects_confluence_threshold() -> None:
    ts = datetime(2024, 1, 1)
    agent = DecisionAgent(config=DecisionConfig(min_confluence=1.4))

    decision, votes, reason = agent.decide(ts, tech_signal=1, value=0.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {},
        "below_min_confluence",
    )

    time_model = TimeOnlyModel({0: (-1.0, 1.0)}, q_low=-1.0, q_high=1.0)
    agent2 = DecisionAgent(config=DecisionConfig(time_model=time_model, min_confluence=1.4))
    decision, votes, reason = agent2.decide(ts, tech_signal=1, value=2.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "BUY",
        {"tech": 1, "time": 1},
        "ok",
    )


def test_decision_agent_accepts_mapping_and_dataclass() -> None:
    ts = datetime(2024, 1, 1)
    agent = DecisionAgent()

    mapping_signal = {"action": 1, "technical_score": 1.0, "confidence_tech": 1.0}
    res = agent.decide(ts, tech_signal=mapping_signal, value=0.0, symbol="EURUSD")
    assert (res.decision, res.votes, res.reason, res.weight) == (
        "BUY",
        {"tech": 1},
        "ok",
        0.9,
    )

    sig = TechnicalSignal(action=-1, technical_score=-1.5, confidence_tech=1.0)
    res2 = agent.decide(ts, tech_signal=sig, value=0.0, symbol="EURUSD")
    assert (res2.decision, res2.votes, res2.reason, res2.weight) == (
        "SELL",
        {"tech": -1},
        "ok",
        -0.9,
    )


def test_decision_agent_accepts_string_actions() -> None:
    ts = datetime(2024, 1, 1)
    agent = DecisionAgent()

    mapping_signal = {"action": "BUY", "technical_score": 2.0, "confidence_tech": 1.0}
    res = agent.decide(ts, tech_signal=mapping_signal, value=0.0, symbol="EURUSD")
    assert (res.decision, res.votes, res.reason, res.weight) == (
        "BUY",
        {"tech": 1},
        "ok",
        0.9,
    )

    sig = TechnicalSignal(action="SELL", technical_score=-3.0, confidence_tech=1.0)
    res2 = agent.decide(ts, tech_signal=sig, value=0.0, symbol="EURUSD")
    assert (res2.decision, res2.votes, res2.reason, res2.weight) == (
        "SELL",
        {"tech": -1},
        "ok",
        -0.9,
    )


@dataclass
class _SimpleSignal:
    action: str
    technical_score: float = 1.0
    confidence_tech: float = 1.0


def test_decision_agent_handles_custom_dataclass_with_string_action() -> None:
    ts = datetime(2024, 1, 1)
    agent = DecisionAgent()
    sig = _SimpleSignal(action="BUY")
    res = agent.decide(ts, tech_signal=sig, value=0.0, symbol="EURUSD")
    assert (res.decision, res.votes, res.reason, res.weight) == (
        "BUY",
        {"tech": 1},
        "ok",
        0.9,
    )
