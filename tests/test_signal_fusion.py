from datetime import datetime
from typing import Literal

import pytest

from forest5.backtest.engine import _fuse_with_time
from forest5.decision import DecisionAgent, DecisionConfig
from forest5.signals.fusion import fuse_signals


class DummyTimeModel:
    def __init__(self, decision: Literal["BUY", "SELL", "WAIT"]) -> None:
        self.decision = decision

    def decide(self, ts, value: float) -> Literal["BUY", "SELL", "WAIT"]:  # pragma: no cover - simple
        return self.decision


def _map(dec: int) -> str:
    return "BUY" if dec > 0 else "SELL" if dec < 0 else "WAIT"


@pytest.mark.parametrize(
    ("tech", "time_sig", "min_conf", "ai", "expected"),
    [
        (1, "WAIT", 2, None, 0),
        (1, "BUY", 2, None, 1),
        (1, "SELL", 1, None, 0),
        (-1, None, 1, None, -1),
        (1, None, 2, 1, 1),
        (1, None, 1, -1, 0),
    ],
)
def test_fusion_matches_engine(tech, time_sig, min_conf, ai, expected) -> None:
    ts = datetime(2024, 1, 1)
    tm = DummyTimeModel(time_sig) if time_sig else None
    fused, _, _ = fuse_signals(tech, ts, 1.0, tm, min_conf, ai)
    engine = _fuse_with_time(tech, ts, 1.0, tm, min_conf, ai)
    assert fused == engine == expected


@pytest.mark.parametrize(
    ("time_sig", "tech_sig", "value"),
    [
        ("WAIT", 1, 0.0),
        ("BUY", 1, 2.0),
        ("SELL", 1, 2.0),
    ],
)
def test_decision_agent_consistency(time_sig, tech_sig, value) -> None:
    ts = datetime(2024, 1, 1)
    tm = DummyTimeModel(time_sig)
    agent = DecisionAgent(config=DecisionConfig(time_model=tm))

    agent_decision, agent_votes, agent_reason = agent.decide(
        ts, tech_signal=tech_sig, value=value, symbol="EURUSD"
    )
    fused_decision, votes, reason = fuse_signals(tech_sig, ts, value, tm)
    assert agent_decision == _map(fused_decision)
    assert agent_votes == votes
    assert agent_reason == reason

