from __future__ import annotations

from forest5.ai_agent import SentimentAgent, Sentiment


class DummyOpenAIClient:
    def __init__(self) -> None:
        self.captured = ""
        self.ChatCompletion = self

    def create(self, model: str, messages: list[dict], max_tokens: int):
        self.captured = messages[0]["content"]
        return {"choices": [{"message": {"content": '{"score": 1, "reason": "ok"}'}}]}


def test_analyse_injects_symbol() -> None:
    client = DummyOpenAIClient()
    agent = SentimentAgent()
    agent.enabled = True
    agent._client = client
    agent._mode = "legacy"

    symbol = "EURUSD"
    ctx = "some context"
    result = agent.analyse(ctx, symbol)

    assert symbol in client.captured
    assert isinstance(result, Sentiment)


def test_decision_agent_passes_symbol(monkeypatch) -> None:
    from forest5.decision import DecisionAgent, DecisionConfig

    class DummyAI:
        def __init__(self) -> None:
            self.args: tuple[str, str] | None = None

        def analyse(self, context: str, symbol: str) -> Sentiment:
            self.args = (context, symbol)
            return Sentiment(0, "")

    config = DecisionConfig(use_ai=True)
    agent = DecisionAgent(config=config)
    dummy = DummyAI()
    agent.ai = dummy

    agent.decide(0, 1, symbol="EURUSD", context_text="ctx")
    assert dummy.args == ("ctx", "EURUSD")


class DummyBadDataClient:
    def __init__(self) -> None:
        self.ChatCompletion = self

    def create(self, model: str, messages: list[dict], max_tokens: int):
        return {"choices": [{"message": {"content": '{"score": "bad", "reason": 123}'}}]}


def test_analyse_handles_invalid_payload() -> None:
    client = DummyBadDataClient()
    agent = SentimentAgent()
    agent.enabled = True
    agent._client = client
    agent._mode = "legacy"

    result = agent.analyse("ctx", "EURUSD")

    assert result.score == 0
    assert result.reason == ""


class DummyBadJSONClient:
    def __init__(self) -> None:
        self.ChatCompletion = self

    def create(self, model: str, messages: list[dict], max_tokens: int):
        return {"choices": [{"message": {"content": "not json"}}]}


def test_analyse_handles_malformed_json() -> None:
    client = DummyBadJSONClient()
    agent = SentimentAgent()
    agent.enabled = True
    agent._client = client
    agent._mode = "legacy"

    result = agent.analyse("ctx", "EURUSD")

    assert result.score == 0
    assert "AI error" in result.reason
