from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import RiskSettings, AISettings
from .config.strategy import BaseStrategySettings
from .utils.timeframes import normalize_timeframe


class StrategySettings(BaseStrategySettings):
    timeframe: str = "1m"

    @field_validator("timeframe")
    @classmethod
    def _norm_tf(cls, v: str) -> str:
        return normalize_timeframe(v)


class BrokerSettings(BaseModel):
    type: str = "mt4"
    bridge_dir: Path | None = None
    symbol: str = "SYMBOL"
    volume: float = 1.0
    timeframe: str = "1m"

    @field_validator("bridge_dir", mode="before")
    @classmethod
    def _to_path(cls, v: str | Path | None) -> Path | None:
        if v in (None, ""):
            return None
        return Path(v)


class DecisionSettings(BaseModel):
    min_confluence: float = 1.0


class LiveTimeModelSettings(BaseModel):
    enabled: bool = False
    path: Path | None = None
    q_low: float = 0.1
    q_high: float = 0.9

    @field_validator("path", mode="before")
    @classmethod
    def _to_path(cls, v: str | Path | None) -> Path | None:
        if v in (None, ""):
            return None
        return Path(v)

    @model_validator(mode="after")
    def _check_quantiles(self) -> "LiveTimeModelSettings":
        if not (0.0 <= self.q_low < self.q_high <= 1.0):
            raise ValueError("0.0 <= q_low < q_high <= 1.0")
        return self


class PatternToggle(BaseModel):
    enabled: bool = True


class PrimaryPatternsSettings(BaseModel):
    engulf: PatternToggle = Field(default_factory=PatternToggle)
    pinbar: PatternToggle = Field(default_factory=PatternToggle)
    star: PatternToggle = Field(default_factory=PatternToggle)


class PrimarySignalSettings(BaseModel):
    strategy: BaseStrategySettings = Field(default_factory=BaseStrategySettings)
    patterns: PrimaryPatternsSettings = Field(default_factory=PrimaryPatternsSettings)


class LiveTimeSettings(BaseModel):
    model: LiveTimeModelSettings = Field(default_factory=LiveTimeModelSettings)
    primary_signal: PrimarySignalSettings = Field(default_factory=PrimarySignalSettings)


class LiveSettings(BaseModel):
    broker: BrokerSettings
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    ai: AISettings = Field(default_factory=AISettings)
    decision: DecisionSettings = Field(default_factory=DecisionSettings)
    time: LiveTimeSettings = Field(default_factory=LiveTimeSettings)


__all__ = [
    "BrokerSettings",
    "StrategySettings",
    "DecisionSettings",
    "LiveTimeModelSettings",
    "PatternToggle",
    "PrimaryPatternsSettings",
    "PrimarySignalSettings",
    "LiveTimeSettings",
    "LiveSettings",
]
