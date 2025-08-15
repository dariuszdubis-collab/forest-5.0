from __future__ import annotations

from dataclasses import dataclass

from .ai_agent import SentimentAgent
from .numerology import NumerologyRules, is_trade_allowed
from .live.router import OrderRouter, PaperBroker


@dataclass
class DecisionConfig:
    use_ai: bool = False
    numerology: NumerologyRules = NumerologyRules()
    min_confluence: int = 1  # ile sygnałów min. (techniczny zawsze liczymy jako 1)


class DecisionAgent:
    """
    Minimalna fuzja: sygnał techniczny (+1/-1/0) + AI (+1/0/-1) + numerologia -> decyzja BUY/SELL/WAIT.
    Ten agent jest szkieletem pod tryb live; w backteście używaj backtest.engine.
    """

    def __init__(self, router: OrderRouter | None = None, config: DecisionConfig | None = None) -> None:
        self.router = router or PaperBroker()
        self.config = config or DecisionConfig()
        self.ai = SentimentAgent() if self.config.use_ai else None

    def decide(self, ts, tech_signal: int, context_text: str = "") -> str:
        # filtr numerologiczny
        if not is_trade_allowed(ts, self.config.numerology):
            return "WAIT"

        votes = [tech_signal]
        if self.ai:
            s = self.ai.analyse(context_text).score
            votes.append(s)

        score = sum(1 if v > 0 else (-1 if v < 0 else 0) for v in votes)
        if score > 0:
            return "BUY"
        if score < 0:
            return "SELL"
        return "WAIT"

