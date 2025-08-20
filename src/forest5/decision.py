from __future__ import annotations

from dataclasses import dataclass

from .ai_agent import SentimentAgent
from .live.router import OrderRouter, PaperBroker
from .time_only import TimeOnlyModel


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
        ts,
        tech_signal: int,
        value: float,
        symbol: str,
        context_text: str = "",
    ) -> tuple[str, dict[str, int], str]:
        votes = {
            "tech": 1 if tech_signal > 0 else (-1 if tech_signal < 0 else 0),
            "time": 0,
            "ai": 0,
        }

        if self.config.time_model:
            tm_decision = self.config.time_model.decide(ts, value)
            if tm_decision == "WAIT":
                return "WAIT", votes, "time_wait"
            votes["time"] = 1 if tm_decision == "BUY" else -1

        if self.ai:
            s = self.ai.analyse(context_text, symbol).score
            votes["ai"] = 1 if s > 0 else (-1 if s < 0 else 0)

        pos = sum(1 for v in votes.values() if v > 0)
        neg = sum(1 for v in votes.values() if v < 0)

        if max(pos, neg) < (self.config.min_confluence or 1):
            return "WAIT", votes, "no_consensus"

        if pos > neg:
            return "BUY", votes, "buy_majority"
        if neg > pos:
            return "SELL", votes, "sell_majority"
        return "WAIT", votes, "no_consensus"
