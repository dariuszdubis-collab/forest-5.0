from __future__ import annotations

from typing import Literal


def _to_sign(value: int | float) -> int:
    """Convert numeric values to ``-1``, ``0`` or ``1``."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def fuse_signals(
    tech_sig: int,
    time_sig: Literal["BUY", "SELL", "WAIT"] | None = None,
    *,
    ai: int | None = None,
    min_conf: int = 1,
) -> tuple[int, str]:
    """Fuse technical, time and optional AI signals into a single decision.

    Parameters
    ----------
    tech_sig:
        Technical signal: ``-1`` (sell), ``0`` (neutral) or ``1`` (buy).
    time_sig:
        Time model decision: ``"BUY"``, ``"SELL"`` or ``"WAIT"``. ``None``
        skips time-based voting.
    ai:
        Optional AI vote. Any non-``None`` value participates in the vote.
    min_conf:
        Minimum number of positive or negative votes required to produce a
        directional decision.

    Returns
    -------
    tuple[int, str]
        The fused signal (``-1``/``0``/``1``) and a short reason string.
    """

    votes = {"tech": _to_sign(tech_sig), "time": 0, "ai": 0}
    pos = {"tech": int(votes["tech"] > 0), "time": 0, "ai": 0}
    neg = {"tech": int(votes["tech"] < 0), "time": 0, "ai": 0}

    if time_sig is not None:
        if time_sig == "WAIT":
            return 0, "time_wait"
        votes["time"] = _to_sign(1 if time_sig == "BUY" else -1)
        pos["time"] = int(votes["time"] > 0)
        neg["time"] = int(votes["time"] < 0)

    if ai is not None:
        votes["ai"] = _to_sign(ai)
        pos["ai"] = int(votes["ai"] > 0)
        neg["ai"] = int(votes["ai"] < 0)

    pos_total = sum(pos.values())
    neg_total = sum(neg.values())
    if max(pos_total, neg_total) < max(min_conf, 1):
        return 0, "no_consensus"
    if pos_total > neg_total:
        return 1, "buy_majority"
    if neg_total > pos_total:
        return -1, "sell_majority"
    return 0, "no_consensus"
