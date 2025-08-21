from datetime import datetime
from typing import Literal

from forest5.backtest.engine import _fuse_with_time


class DummyTimeModel:
    def __init__(self, decision: Literal["BUY", "SELL", "WAIT"]) -> None:
        self.decision = decision

    def decide(self, ts, value: float) -> Literal["BUY", "SELL", "WAIT"]:
        return self.decision


def test_wait_short_circuit() -> None:
    tm = DummyTimeModel("WAIT")
    ts = datetime(2024, 1, 1)
    assert _fuse_with_time(1, ts, 1.0, tm, 2) == 0


def test_confluence_requires_both_votes() -> None:
    tm = DummyTimeModel("BUY")
    ts = datetime(2024, 1, 1)
    assert _fuse_with_time(1, ts, 1.0, tm, 2) == 1


def test_conflict_returns_neutral() -> None:
    tm = DummyTimeModel("SELL")
    ts = datetime(2024, 1, 1)
    assert _fuse_with_time(1, ts, 1.0, tm, 1) == 0


def test_passthrough_without_time_model() -> None:
    ts = datetime(2024, 1, 1)
    assert _fuse_with_time(-1, ts, 1.0, None, 1) == -1
