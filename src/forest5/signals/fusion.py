from __future__ import annotations

"""Utilities for fusing technical, time and AI signals."""

from typing import Dict, Tuple

import pandas as pd

from ..time_only import TimeOnlyModel


def fuse_signals(
    tech_signal: int,
    ts: pd.Timestamp,
    value: float,
    time_model: TimeOnlyModel | None = None,
    min_conf: int = 1,
    ai_decision: int | None = None,
) -> Tuple[int, Dict[str, int], str]:
    """Combine votes from technical, time and AI models.

    Parameters
    ----------
    tech_signal:
        Raw technical signal, typically -1/0/1.
    ts:
        Timestamp of the observation.
    value:
        Current price/value used by the time model.
    time_model:
        Optional time-of-day model producing BUY/SELL/WAIT decisions.
    min_conf:
        Minimum number of agreeing votes required for BUY/SELL.
    ai_decision:
        Optional AI vote (-1/0/1).

    Returns
    -------
    tuple
        ``(decision, votes, reason)`` where ``decision`` is -1/0/1.
    """

    votes = {
        "tech": 1 if tech_signal > 0 else (-1 if tech_signal < 0 else 0),
        "time": 0,
        "ai": 0,
    }
    pos = {"tech": 1 if votes["tech"] > 0 else 0, "time": 0, "ai": 0}
    neg = {"tech": 1 if votes["tech"] < 0 else 0, "time": 0, "ai": 0}

    if time_model:
        tm_decision = time_model.decide(ts, value)
        if tm_decision == "WAIT":
            return 0, votes, "time_wait"
        votes["time"] = 1 if tm_decision == "BUY" else -1
        pos["time"] = 1 if votes["time"] > 0 else 0
        neg["time"] = 1 if votes["time"] < 0 else 0

    if ai_decision is not None:
        votes["ai"] = 1 if ai_decision > 0 else (-1 if ai_decision < 0 else 0)
        pos["ai"] = 1 if votes["ai"] > 0 else 0
        neg["ai"] = 1 if votes["ai"] < 0 else 0

    pos_total = sum(pos.values())
    neg_total = sum(neg.values())
    if max(pos_total, neg_total) < max(min_conf, 1):
        return 0, votes, "no_consensus"
    if pos_total > neg_total:
        return 1, votes, "buy_majority"
    if neg_total > pos_total:
        return -1, votes, "sell_majority"
    return 0, votes, "no_consensus"


__all__ = ["fuse_signals"]

