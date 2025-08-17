from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Sentiment:
    score: int  # -1 / 0 / +1
    reason: str


class SentimentAgent:
    """
    Uproszczony agent.
    Jeśli brak OPENAI_API_KEY lub brak pakietu openai -> zwraca neutral.
    """

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 256) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.enabled = bool(os.getenv("OPENAI_API_KEY"))

        try:
            # próba nowego SDK
            from openai import OpenAI  # type: ignore

            self._client = OpenAI()
            self._mode = "sdk1"
        except Exception:
            try:
                import openai  # type: ignore

                self._client = openai
                self._client.api_key = os.getenv("OPENAI_API_KEY")
                self._mode = "legacy"
            except Exception:
                self.enabled = False
                self._client = None
                self._mode = "none"

    def analyse(self, context: str, symbol: str) -> Sentiment:
        if not self.enabled or self._client is None:
            return Sentiment(0, "AI disabled or no key; neutral filter.")

        prompt = (
            "Determine market sentiment "
            "(-1 bearish / 0 neutral / +1 bullish). "
            "Symbol: {symbol}. Context:\n{ctx}\n"
            "Answer as JSON with keys: score, reason."
        )

        try:
            if self._mode == "sdk1":
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt.format(
                                symbol=symbol,
                                ctx=context,
                            ),
                        }
                    ],
                    max_tokens=self.max_tokens,
                )
                txt = resp.choices[0].message.content or ""
            else:
                txt = self._client.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt.format(
                                symbol=symbol,
                                ctx=context,
                            ),
                        }
                    ],
                    max_tokens=self.max_tokens,
                )["choices"][0]["message"]["content"]

            # bardzo defensywne parsowanie
            import json

            data = json.loads(txt)
            score = int(data.get("score", 0))
            reason = str(data.get("reason", ""))
            score = -1 if score < 0 else (1 if score > 0 else 0)
            return Sentiment(score, reason[:200])
        except Exception as e:
            return Sentiment(0, f"AI error -> neutral: {e}")
