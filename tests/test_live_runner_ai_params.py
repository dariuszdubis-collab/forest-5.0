from pathlib import Path

from forest5.live.live_runner import run_live
from forest5.live.settings import (
    LiveSettings,
    BrokerSettings,
    DecisionSettings,
    AISettings,
    TimeSettings,
    RiskSettings,
)


class DummySentimentAgent:
    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
        self.enabled = True

    def analyse(self, context: str, symbol: str):
        from forest5.ai_agent import Sentiment

        return Sentiment(0, "")


def test_run_live_passes_ai_params(tmp_path: Path, monkeypatch):
    captured: dict[str, int | str] = {}

    def _sentiment_agent(model: str, max_tokens: int):
        captured["model"] = model
        captured["max_tokens"] = max_tokens
        return DummySentimentAgent(model, max_tokens)

    monkeypatch.setattr("forest5.decision.SentimentAgent", _sentiment_agent)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    ctx = tmp_path / "ctx.txt"
    ctx.write_text("hi", encoding="utf-8")
    settings = LiveSettings(
        broker=BrokerSettings(type="paper", bridge_dir=str(tmp_path), symbol="EURUSD", volume=0.01),
        decision=DecisionSettings(min_confluence=0.5),
        ai=AISettings(enabled=True, model="custom-model", max_tokens=99, context_file=str(ctx), require_context=False),
        time=TimeSettings(),
        risk=RiskSettings(max_drawdown=0.5),
    )

    run_live(settings, max_steps=0, timeout=0)

    assert captured == {"model": "custom-model", "max_tokens": 99}
