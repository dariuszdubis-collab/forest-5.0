from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ai_agent import SentimentAgent
from .live.router import OrderRouter, PaperBroker
from .time_only import TimeOnlyModel
from .signals.fusion import fuse_signals


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
        ai_vote: int | None = None
        if self.ai:
            s = self.ai.analyse(context_text, symbol).score
            ai_vote = 1 if s > 0 else (-1 if s < 0 else 0)

        decision, votes, reason = fuse_signals(
            tech_signal,
            ts,
            value,
            self.config.time_model,
            self.config.min_confluence,
            ai_vote,
        )

        if decision > 0:
            return "BUY", votes, reason
        if decision < 0:
            return "SELL", votes, reason
        return "WAIT", votes, reason
