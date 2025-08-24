from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Mapping

import pandas as pd

from .ai_agent import SentimentAgent
from .live.router import OrderRouter, PaperBroker
from .time_only import TimeOnlyModel
from .signals.fusion import _to_sign


def _normalize_action(action: int | float | str) -> int | float:
    """Convert common string actions to numeric values."""
    if isinstance(action, str):
        return {"BUY": 1, "SELL": -1, "KEEP": 0, "WAIT": 0}.get(action.upper(), 0)
    return action


@dataclass
class DecisionResult:
    """Result of a decision fusion.

    ``decision`` is one of ``"BUY"``, ``"SELL"`` or ``"WAIT"``. ``weight``
    denotes the confidence in the decision (0–1). ``votes`` keep the raw sign
    of each component (technical/time/AI) for backwards compatibility.

    The class implements ``__iter__`` so existing code expecting a 3-tuple
    ``(decision, votes, reason)`` continues to work.
    """

    decision: str
    weight: float
    votes: dict[str, int]
    reason: str

    def __iter__(self):  # pragma: no cover - tiny helper
        yield self.decision
        yield self.votes
        yield self.reason


@dataclass
class DecisionConfig:
    use_ai: bool = False
    ai_model: str = "gpt-4o-mini"
    ai_max_tokens: int = 256
    time_model: TimeOnlyModel | None = None
    # Threshold of combined weights required to take a trade.
    # Technical signal always contributes at least 1.
    min_confluence: float = 1.0


class DecisionAgent:
    """
    Minimalna fuzja: sygnał techniczny (+1/-1/0) + AI (+1/0/-1) + model czasowy (+1/-1)
    -> decyzja BUY/SELL/WAIT.
    Ten agent jest szkieletem pod tryb live;
    w backteście używaj backtest.engine.
    """

    def __init__(
        self,
        router: OrderRouter | None = None,
        config: DecisionConfig | None = None,
    ) -> None:
        self.router = router or PaperBroker()
        self.config = config or DecisionConfig()
        self.ai = (
            SentimentAgent(model=self.config.ai_model, max_tokens=self.config.ai_max_tokens)
            if self.config.use_ai
            else None
        )

    def decide(
        self,
        ts: pd.Timestamp,
        tech_signal: int | Mapping[str, object],
        value: float,
        symbol: str,
        context_text: str = "",
    ) -> DecisionResult:
        if isinstance(tech_signal, Mapping) or is_dataclass(tech_signal):
            get = (
                tech_signal.get
                if isinstance(tech_signal, Mapping)
                else lambda k, d=None: getattr(tech_signal, k, d)
            )
            action = _normalize_action(get("action", 0))
            tech_score = get("technical_score", action)
            confidence = get("confidence_tech", 1.0)
            votes: dict[str, tuple[int, float]] = {
                "tech": (
                    _to_sign(action if action else tech_score),
                    abs(float(tech_score)) * float(confidence),
                ),
                "time": (0, 0.0),
                "ai": (0, 0.0),
            }
        else:
            votes: dict[str, tuple[int, float]] = {
                "tech": (_to_sign(int(tech_signal)), 1.0),
                "time": (0, 0.0),
                "ai": (0, 0.0),
            }

        if self.config.time_model:
            tm_res = self.config.time_model.decide(ts, value)
            if isinstance(tm_res, tuple):
                tm_decision, tm_weight = tm_res
            else:  # backward compatibility
                tm_decision, tm_weight = tm_res, 1.0
            if tm_decision == "WAIT":
                return DecisionResult("WAIT", 0.0, {k: v[0] for k, v in votes.items()}, "time_wait")
            votes["time"] = (_to_sign(1 if tm_decision == "BUY" else -1), float(tm_weight))

        if self.ai:
            s = self.ai.analyse(context_text, symbol).score
            votes["ai"] = (_to_sign(s), abs(float(s)))

        pos_total = sum(w for s, w in votes.values() if s > 0)
        neg_total = sum(w for s, w in votes.values() if s < 0)
        if max(pos_total, neg_total) < max(self.config.min_confluence, 1.0):
            return DecisionResult(
                "WAIT",
                0.0,
                {k: v[0] for k, v in votes.items()},
                "no_consensus",
            )

        if pos_total > neg_total:
            dec = "BUY"
            weights = [w for s, w in votes.values() if s > 0]
        elif neg_total > pos_total:
            dec = "SELL"
            weights = [w for s, w in votes.values() if s < 0]
        else:
            return DecisionResult(
                "WAIT",
                0.0,
                {k: v[0] for k, v in votes.items()},
                "no_consensus",
            )

        final_weight = min(weights) if weights else 0.0
        reason = "buy_majority" if dec == "BUY" else "sell_majority"
        return DecisionResult(dec, final_weight, {k: v[0] for k, v in votes.items()}, reason)
