from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal, Mapping

import pandas as pd

from .ai_agent import SentimentAgent
from .live.router import OrderRouter, PaperBroker
from .time_only import TimeOnlyModel
from .signals.fusion import _to_sign


Dir = Literal[-1, 0, 1]


@dataclass
class DecisionVote:
    source: str
    direction: Dir
    weight: float = 0.0
    score: float = 0.0
    meta: dict[str, Any] | None = None


@dataclass(init=False)
class DecisionResult:
    action: Literal["BUY", "SELL", "WAIT"]
    weight_sum: float
    reason: str = ""
    details: dict[str, Any] | None = None

    def __init__(
        self,
        action: Literal["BUY", "SELL", "WAIT"],
        weight_sum: float,
        reason_or_details: str | dict[str, Any] = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(reason_or_details, dict):
            self.action = action
            self.weight_sum = weight_sum
            self.details = reason_or_details
            self.reason = details or ""
        else:
            self.action = action
            self.weight_sum = weight_sum
            self.reason = reason_or_details
            self.details = details or {}

    # Backward compatibility helpers
    @property
    def decision(self) -> Literal["BUY", "SELL", "WAIT"]:
        return self.action

    @property
    def weight(self) -> float:
        return self.weight_sum

    @property
    def votes(self) -> dict[str, Dir]:
        if not self.details:
            return {}
        out: dict[str, Dir] = {}
        for k, v in self.details.items():
            if isinstance(v, Mapping) and "direction" in v:
                out[k] = v["direction"]  # type: ignore[assignment]
            else:
                out[k] = v  # type: ignore[assignment]
        return out

    def __iter__(self):  # pragma: no cover - tiny helper
        yield self.action
        yield self.votes
        yield self.reason


def _normalize_action(action: int | float | str) -> int | float:
    """Convert common string actions to numeric values."""
    if isinstance(action, str):
        return {"BUY": 1, "SELL": -1, "KEEP": 0, "WAIT": 0}.get(action.upper(), 0)
    return action


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
            votes: dict[str, DecisionVote] = {
                "tech": DecisionVote(
                    source="tech",
                    direction=_to_sign(action if action else tech_score),
                    weight=abs(float(tech_score)) * float(confidence),
                    score=float(tech_score),
                ),
                "time": DecisionVote(source="time", direction=0),
                "ai": DecisionVote(source="ai", direction=0),
            }
        else:
            votes = {
                "tech": DecisionVote(
                    source="tech",
                    direction=_to_sign(int(tech_signal)),
                    weight=1.0,
                    score=float(tech_signal),
                ),
                "time": DecisionVote(source="time", direction=0),
                "ai": DecisionVote(source="ai", direction=0),
            }

        if self.config.time_model:
            tm_res = self.config.time_model.decide(ts, value)
            if isinstance(tm_res, tuple):
                tm_decision, tm_weight = tm_res
            else:  # backward compatibility
                tm_decision, tm_weight = tm_res, 1.0
            if tm_decision == "WAIT":
                return DecisionResult(
                    "WAIT",
                    0.0,
                    "time_wait",
                    {k: asdict(v) for k, v in votes.items()},
                )
            votes["time"] = DecisionVote(
                source="time",
                direction=_to_sign(1 if tm_decision == "BUY" else -1),
                weight=float(tm_weight),
                score=float(tm_weight),
            )

        if self.ai:
            s = self.ai.analyse(context_text, symbol).score
            votes["ai"] = DecisionVote(
                source="ai",
                direction=_to_sign(s),
                weight=abs(float(s)),
                score=float(s),
            )

        pos_total = sum(v.weight for v in votes.values() if v.direction > 0)
        neg_total = sum(v.weight for v in votes.values() if v.direction < 0)
        if max(pos_total, neg_total) < max(self.config.min_confluence, 1.0):
            return DecisionResult(
                "WAIT",
                0.0,
                "no_consensus",
                {k: asdict(v) for k, v in votes.items()},
            )

        if pos_total > neg_total:
            dec = "BUY"
            weights = [v.weight for v in votes.values() if v.direction > 0]
        elif neg_total > pos_total:
            dec = "SELL"
            weights = [v.weight for v in votes.values() if v.direction < 0]
        else:
            return DecisionResult(
                "WAIT",
                0.0,
                "no_consensus",
                {k: asdict(v) for k, v in votes.items()},
            )

        final_weight = min(weights) if weights else 0.0
        reason = "buy_majority" if dec == "BUY" else "sell_majority"
        return DecisionResult(
            dec,
            final_weight,
            reason,
            {k: asdict(v) for k, v in votes.items()},
        )
