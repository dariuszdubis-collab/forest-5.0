from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..decision import DecisionResult, DecisionVote


def _to_sign(value: int | float) -> int:
    """Convert numeric values to -1, 0 or 1."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _fuse_votes(votes: list["DecisionVote"], cfg) -> "DecisionResult":
    from ..decision import DecisionResult

    s = sum(v.direction * v.weight for v in votes)
    min_conf = getattr(cfg.decision, "min_confluence", 0.0)
    tie_eps = getattr(cfg.decision, "tie_epsilon", 0.05)
    if abs(s) < tie_eps:
        return DecisionResult("WAIT", s, "tie")
    if abs(s) < min_conf:
        return DecisionResult("WAIT", s, "below_min_confluence")
    action = "BUY" if s > 0 else "SELL"
    return DecisionResult(
        action,
        s,
        "ok",
        {"votes": [v.__dict__ for v in votes]},
    )
