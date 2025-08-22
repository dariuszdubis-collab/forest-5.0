from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ai_agent import SentimentAgent
from .live.router import OrderRouter, PaperBroker
from .time_only import TimeOnlyModel
from .signals.fusion import _to_sign


@dataclass
class DecisionConfig:
    use_ai: bool = False
    time_model: TimeOnlyModel | None = None
    min_confluence: int = 1  # min sygnałów (techniczny zawsze = 1)


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
        self.ai = SentimentAgent() if self.config.use_ai else None

    def decide(
        self,
        ts: pd.Timestamp,
        tech_signal: int,
        value: float,
        symbol: str,
        context_text: str = "",
    ) -> tuple[str, dict[str, int], str]:
        votes = {"tech": _to_sign(tech_signal), "time": 0, "ai": 0}
        pos = {"tech": int(votes["tech"] > 0), "time": 0, "ai": 0}
        neg = {"tech": int(votes["tech"] < 0), "time": 0, "ai": 0}

        if self.config.time_model:
            tm_decision = self.config.time_model.decide(ts, value)
            if tm_decision == "WAIT":
                return "WAIT", votes, "time_wait"
            votes["time"] = _to_sign(1 if tm_decision == "BUY" else -1)
            pos["time"] = int(votes["time"] > 0)
            neg["time"] = int(votes["time"] < 0)

        if self.ai:
            s = self.ai.analyse(context_text, symbol).score
            votes["ai"] = _to_sign(s)
            pos["ai"] = int(votes["ai"] > 0)
            neg["ai"] = int(votes["ai"] < 0)

        pos_total = sum(pos.values())
        neg_total = sum(neg.values())
        if max(pos_total, neg_total) < (self.config.min_confluence or 1):
            return "WAIT", votes, "no_consensus"

        if pos_total > neg_total:
            return "BUY", votes, "buy_majority"
        if neg_total > pos_total:
            return "SELL", votes, "sell_majority"
        return "WAIT", votes, "no_consensus"
