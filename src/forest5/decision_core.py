from __future__ import annotations

from typing import Tuple

import pandas as pd

from .time_only import TimeOnlyModel
from .signals.fusion import _to_sign


def fuse_with_time(
    tech_signal: int,
    ts: pd.Timestamp,
    price: float,
    time_model: TimeOnlyModel | None,
    min_conf: float,
    ai_decision: int | tuple[int, float] | None = None,
    *,
    return_reason: bool = False,
    return_weight: bool = False,
):
    """Fuse technical, time, and optional AI signals into one decision.

    Returns -1/0/1 or tuples depending on flags, mirroring the previous
    `_fuse_with_time` helper used in backtest engine.
    """

    votes: dict[str, tuple[int, float]] = {
        "tech": (_to_sign(tech_signal), 1.0),
        "time": (0, 0.0),
        "ai": (0, 0.0),
    }

    if time_model:
        tm_res = time_model.decide(ts)
        if isinstance(tm_res, dict):
            tm_decision = tm_res.get("decision")
            tm_weight = tm_res.get("confidence", 1.0)
        elif isinstance(tm_res, tuple):
            tm_decision, tm_weight = tm_res
        else:
            tm_decision, tm_weight = tm_res, 1.0
        if tm_decision in {"WAIT", "HOLD"}:
            reason = "time_model_hold" if tm_decision == "HOLD" else "time_model_wait"
            if return_reason and return_weight:
                return 0, 0.0, reason
            if return_reason:
                return 0, reason
            if return_weight:
                return 0, 0.0
            return 0
        votes["time"] = (_to_sign(1 if tm_decision == "BUY" else -1), float(tm_weight))

    if ai_decision is not None:
        if isinstance(ai_decision, tuple):
            ai_sig, ai_weight = ai_decision
        else:
            ai_sig, ai_weight = ai_decision, 1.0
        votes["ai"] = (_to_sign(ai_sig), float(ai_weight))

    pos_total = sum(w for s, w in votes.values() if s > 0)
    neg_total = sum(w for s, w in votes.values() if s < 0)
    if max(pos_total, neg_total) < max(min_conf, 1.0):
        if return_reason and return_weight:
            return 0, 0.0, "not_enough_confluence"
        if return_reason:
            return 0, "not_enough_confluence"
        if return_weight:
            return 0, 0.0
        return 0

    if pos_total > neg_total:
        res_dec = 1
        weights = [w for s, w in votes.values() if s > 0]
    elif neg_total > pos_total:
        res_dec = -1
        weights = [w for s, w in votes.values() if s < 0]
    else:
        res_dec = 0
        weights = []

    final_weight = min(weights) if weights else 0.0

    if return_reason and return_weight:
        return res_dec, final_weight, None
    if return_weight:
        return res_dec, final_weight
    if return_reason:
        return res_dec, None
    return res_dec


__all__ = ["fuse_with_time"]

