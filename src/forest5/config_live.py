from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import json
import yaml

from .config import RiskSettings, AISettings, TimeOnlySettings
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
class LiveSettings:
    broker: BrokerSettings
    strategy: StrategySettings = field(default_factory=StrategySettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    ai: AISettings = field(default_factory=AISettings)
    time: TimeOnlySettings = field(default_factory=TimeOnlySettings)

    @classmethod
    def from_dict(cls, data: dict) -> "LiveSettings":
        return cls(
            broker=BrokerSettings(**data.get("broker", {})),
            strategy=StrategySettings(**data.get("strategy", {})),
            risk=RiskSettings(**data.get("risk", {})),
            ai=AISettings(**data.get("ai", {})),
            time=TimeOnlySettings(**data.get("time", {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "LiveSettings":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() in {".yml", ".yaml"}:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        elif p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise ValueError("Supported: .yaml/.yml/.json")
        return cls.from_dict(data)


__all__ = ["BrokerSettings", "StrategySettings", "LiveSettings"]
