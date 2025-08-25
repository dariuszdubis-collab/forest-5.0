from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from types import SimpleNamespace
from typing import Any, Literal, Mapping

import pandas as pd

from .ai_agent import SentimentAgent
from .live.router import OrderRouter, PaperBroker
from .time_only import TimeOnlyModel
from .signals.fusion import _fuse_votes, _to_sign
from .utils.log import setup_logger


Dir = Literal[-1, 0, 1]


log = setup_logger()


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
        if "votes" in self.details and isinstance(self.details["votes"], list):
            raw_votes = self.details["votes"]
            out: dict[str, Dir] = {}
            for v in raw_votes:
                if isinstance(v, Mapping):
                    src = v.get("source")
                    dir_ = v.get("direction")
                    if isinstance(src, str) and isinstance(dir_, (int, float)):
                        out[src] = _to_sign(dir_)  # type: ignore[assignment]
            return out
        out: dict[str, Dir] = {}
        for k, v in self.details.items():
            if isinstance(v, Mapping) and "direction" in v:
                out[k] = v["direction"]  # type: ignore[assignment]
            else:
                out[k] = v  # type: ignore[assignment]
        return out

    def __iter__(self):  # pragma: no cover - legacy helper
        """Allow unpacking ``decision, votes, reason = result``.

        The second element is a mapping of ``{source: direction}`` extracted from
        the ``details`` field to retain backward compatibility with older
        return formats.
        """
        return iter((self.action, self.votes, self.reason))


def _normalize_action(action: int | float | str) -> int | float:
    """Convert common string actions to numeric values."""
    if isinstance(action, str):
        return {"BUY": 1, "SELL": -1, "KEEP": 0, "WAIT": 0}.get(action.upper(), 0)
    return action


def _normalize_tech_input(tech_signal, cfg) -> DecisionVote:
    """Normalize various technical signal formats into a ``DecisionVote``.

    ``tech_signal`` may be an int/float, mapping, or dataclass. For mappings and
    dataclasses we expect ``action``/``technical_score``/``confidence_tech``
    fields. Confidence is clamped to ``cfg.decision.tech.conf_floor`` and
    ``cfg.decision.tech.conf_cap``. String actions like ``BUY``/``SELL``/``KEEP``
    are mapped to numeric directions. The returned ``DecisionVote`` always
    includes ``meta={'mode': <mode>}`` describing the normalization path.
    Unknown formats yield a neutral vote.
    """

    tech_cfg = getattr(getattr(cfg, "decision", object()), "tech", object())
    default_conf_int = getattr(tech_cfg, "default_conf_int", 0.5)
    conf_floor = getattr(tech_cfg, "conf_floor", 0.2)
    conf_cap = getattr(tech_cfg, "conf_cap", 0.9)
    w_tech = getattr(getattr(getattr(cfg, "decision", object()), "weights", object()), "tech", 1.0)

    def _clamp(v: float) -> float:
        return max(conf_floor, min(conf_cap, v))

    if isinstance(tech_signal, Mapping) or is_dataclass(tech_signal):
        get = (
            tech_signal.get
            if isinstance(tech_signal, Mapping)
            else lambda k, d=None: getattr(tech_signal, k, d)
        )
        action = _normalize_action(get("action", 0))
        tech_score = float(get("technical_score", action))
        conf = _clamp(float(get("confidence_tech", default_conf_int)))
        meta = {}
        raw_meta = get("meta")
        if isinstance(raw_meta, Mapping):
            meta.update(raw_meta)
        meta["mode"] = (
            "dataclass"
            if is_dataclass(tech_signal) and not isinstance(tech_signal, Mapping)
            else "mapping"
        )
        return DecisionVote(
            source="tech",
            direction=_to_sign(action if action else tech_score),
            weight=_clamp(conf * w_tech),
            score=tech_score,
            meta=meta,
        )

    if isinstance(tech_signal, str):
        dir_val = _normalize_action(tech_signal)
        if isinstance(dir_val, (int, float)):
            return DecisionVote(
                source="tech",
                direction=_to_sign(dir_val),
                weight=default_conf_int * w_tech,
                score=float(dir_val),
                meta={"mode": "str"},
            )

    if isinstance(tech_signal, (int, float)):
        return DecisionVote(
            source="tech",
            direction=_to_sign(int(tech_signal)),
            weight=default_conf_int * w_tech,
            score=float(tech_signal),
            meta={"mode": "int"},
        )

    return DecisionVote(source="tech", direction=0, weight=0.0, score=0.0, meta={"mode": "unknown"})


def _normalize_ai_input(ai_signal, cfg) -> DecisionVote:
    """Normalize AI sentiment output into a ``DecisionVote``.

    The function accepts raw numbers, dataclasses, or mappings with ``score``
    and optional ``confidence_ai`` fields.  Missing confidence values default to
    ``cfg.decision.ai.default_conf``.  The vote weight scales with
    ``abs(score)`` and a global AI weight multiplier.
    """

    ai_cfg = getattr(getattr(cfg, "decision", object()), "ai", object())
    tech_cfg = getattr(getattr(cfg, "decision", object()), "tech", object())
    default_conf = getattr(ai_cfg, "default_conf", 1.0)
    conf_floor = getattr(ai_cfg, "conf_floor", getattr(tech_cfg, "conf_floor", 0.2))
    conf_cap = getattr(ai_cfg, "conf_cap", getattr(tech_cfg, "conf_cap", 0.9))
    w_ai = getattr(getattr(getattr(cfg, "decision", object()), "weights", object()), "ai", 1.0)

    def _clamp_ai(v: float) -> float:
        return max(conf_floor, min(conf_cap, v))

    if isinstance(ai_signal, Mapping) or is_dataclass(ai_signal):
        get = (
            ai_signal.get
            if isinstance(ai_signal, Mapping)
            else lambda k, d=None: getattr(ai_signal, k, d)
        )
        score = float(get("score", 0.0))
        conf = float(get("confidence_ai", default_conf))
        mode = (
            "dataclass"
            if is_dataclass(ai_signal) and not isinstance(ai_signal, Mapping)
            else "mapping"
        )
        return DecisionVote(
            source="ai",
            direction=_to_sign(score),
            weight=_clamp_ai(abs(score) * conf * w_ai),
            score=score,
            meta={"mode": mode},
        )

    if isinstance(ai_signal, (int, float)):
        score = float(ai_signal)
        return DecisionVote(
            source="ai",
            direction=_to_sign(score),
            weight=_clamp_ai(abs(score) * default_conf * w_ai),
            score=score,
            meta={"mode": "int"},
        )

    return DecisionVote(source="ai", direction=0, weight=0.0, score=0.0, meta={"mode": "unknown"})


@dataclass
class DecisionConfig:
    use_ai: bool = False
    ai_model: str = "gpt-4o-mini"
    ai_max_tokens: int = 256
    time_model: TimeOnlyModel | None = None
    # Threshold of combined weights required to take a trade.
    # Technical signal always contributes at least 1.
    min_confluence: float = 0.0
    tie_epsilon: float = 0.05
    weights: Any = field(default_factory=lambda: SimpleNamespace(tech=1.0, ai=0.5, time=1.0))
    tech: Any = field(
        default_factory=lambda: SimpleNamespace(default_conf_int=0.5, conf_floor=0.2, conf_cap=0.9)
    )


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
        votes: list[DecisionVote] = []
        cfg = SimpleNamespace(decision=self.config)

        tech_vote = _normalize_tech_input(tech_signal, cfg)
        if tech_vote.direction == 0:
            return DecisionResult("WAIT", 0.0, "no_tech_signal")
        votes.append(tech_vote)

        if self.config.time_model:
            tm_res = self.config.time_model.decide(ts, value)
            if isinstance(tm_res, tuple):
                tm_decision, tm_weight = tm_res
            else:
                tm_decision, tm_weight = tm_res, 1.0
            if tm_decision == "WAIT":
                return DecisionResult("WAIT", 0.0, "timeonly_wait")
            w_time = getattr(getattr(cfg.decision, "weights", object()), "time", 1.0)
            votes.append(
                DecisionVote(
                    source="time",
                    direction=_to_sign(1 if tm_decision == "BUY" else -1),
                    weight=float(w_time),
                    score=float(tm_weight),
                )
            )

        if self.ai:
            ai_sent = self.ai.analyse(context_text, symbol)
            votes.append(_normalize_ai_input(ai_sent, cfg))

        log.debug("decision_inputs", votes=[v.__dict__ for v in votes])
        result = _fuse_votes(votes, cfg)
        log.debug(
            "decision_result",
            action=result.action,
            weight_sum=result.weight_sum,
            reason=result.reason,
        )
        return result
