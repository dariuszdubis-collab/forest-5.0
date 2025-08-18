from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json
import yaml

from .config import RiskSettings, AISettings
from .utils.timeframes import normalize_timeframe


@dataclass
class StrategySettings:
    name: Literal["ema_cross"] = "ema_cross"
    fast: int = 12
    slow: int = 26
    use_rsi: bool = False
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    timeframe: str = "1m"

    def __post_init__(self) -> None:
        self.timeframe = normalize_timeframe(self.timeframe)


@dataclass
class BrokerSettings:
    type: str = "mt4"
    bridge_dir: Path | None = None
    symbol: str = "SYMBOL"
    volume: float = 1.0

    def __post_init__(self) -> None:
        if self.bridge_dir is not None:
            self.bridge_dir = Path(self.bridge_dir)


@dataclass
class DecisionSettings:
    min_confluence: int = 1


@dataclass
class LiveTimeModelSettings:
    enabled: bool = False
    path: Path | None = None
    q_low: float = 0.1
    q_high: float = 0.9

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)
        if not (0.0 <= self.q_low < self.q_high <= 1.0):
            raise ValueError("q_low and q_high must satisfy 0.0 <= q_low < q_high <= 1.0")


@dataclass
class LiveTimeSettings:
    blocked_weekdays: list[int] = field(default_factory=list)
    blocked_hours: list[int] = field(default_factory=list)
    model: LiveTimeModelSettings = field(default_factory=LiveTimeModelSettings)

    def __post_init__(self) -> None:
        invalid_weekdays = [d for d in self.blocked_weekdays if d < 0 or d > 6]
        invalid_hours = [h for h in self.blocked_hours if h < 0 or h > 23]
        if invalid_weekdays:
            raise ValueError(f"blocked_weekdays must be in range 0-6: {invalid_weekdays}")
        if invalid_hours:
            raise ValueError(f"blocked_hours must be in range 0-23: {invalid_hours}")


@dataclass
class LiveSettings:
    broker: BrokerSettings
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    ai: AISettings = field(default_factory=AISettings)
    decision: DecisionSettings = field(default_factory=DecisionSettings)
    time: LiveTimeSettings = field(default_factory=LiveTimeSettings)

    @classmethod
    def from_dict(cls, data: dict) -> "LiveSettings":
        return cls(
            broker=BrokerSettings(**data.get("broker", {})),
            strategy=StrategySettings(**data.get("strategy", {})),
            risk=RiskSettings(**data.get("risk", {})),
            ai=AISettings(**data.get("ai", {})),
            decision=DecisionSettings(**data.get("decision", {})),
            time=LiveTimeSettings(**data.get("time", {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "LiveSettings":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)

        config_dir = p.resolve().parent

        if p.suffix.lower() in {".yml", ".yaml"}:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        elif p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise ValueError("Supported: .yaml/.yml/.json")

        ai_data = data.get("ai", {})
        context_file = ai_data.get("context_file")
        if context_file:
            ai_data["context_file"] = str((config_dir / context_file).resolve())
            data["ai"] = ai_data

        return cls.from_dict(data)


__all__ = [
    "BrokerSettings",
    "StrategySettings",
    "DecisionSettings",
    "LiveTimeModelSettings",
    "LiveTimeSettings",
    "LiveSettings",
]
