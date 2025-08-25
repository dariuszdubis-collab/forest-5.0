from datetime import datetime
from typing import Literal

from forest5.decision import DecisionAgent, DecisionConfig


class DummyTimeModel:
    def decide(
        self, ts, value: float
    ) -> Literal["BUY", "SELL", "WAIT"]:  # pragma: no cover - simple
        return "SELL"


def test_min_confluence_conflicting_votes_wait() -> None:
    """Conflicting votes with min_confluence=2 should result in WAIT."""
    tm = DummyTimeModel()
    cfg = DecisionConfig(time_model=tm, min_confluence=2)
    cfg.weights.time = 0.5
    agent = DecisionAgent(config=cfg)
    ts = datetime(2024, 1, 1)
    decision, votes, reason = agent.decide(ts, tech_signal=1, value=1.0, symbol="EURUSD")
    assert (decision, votes, reason) == (
        "WAIT",
        {},
        "tie",
    )
