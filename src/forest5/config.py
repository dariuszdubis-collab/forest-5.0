from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from .utils.timeframes import normalize_timeframe


class StrategySettings(BaseModel):
    name: Literal["ema_cross"] = "ema_cross"
    fast: int = 12
    slow: int = 26
    use_rsi: bool = False
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30


class RiskSettings(BaseModel):
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    max_drawdown: float = 0.30
    fee_perc: float = 0.0005
    slippage_perc: float = 0.0


class AISettings(BaseModel):
    enabled: bool = False
    model: str = "gpt-4o-mini"
    max_tokens: int = 256


class NumerologySettings(BaseModel):
    enabled: bool = False
    blocked_weekdays: list[int] = Field(default_factory=list)  # 0=Mon..6=Sun
    blocked_hours: list[int] = Field(default_factory=list)     # 0..23


class BacktestSettings(BaseModel):
    symbol: str = "SYMBOL"
    timeframe: str = "1h"
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    atr_period: int = 14
    atr_multiple: float = 2.0

    @field_validator("timeframe")
    @classmethod
    def _norm_tf(cls, v: str) -> str:
        return normalize_timeframe(v)

    @classmethod
    def from_file(cls, path: str | Path) -> "BacktestSettings":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() in {".yml", ".yaml"}:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        elif p.suffix.lower() == ".json":
            import json
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise ValueError("Supported: .yaml/.yml/.json")
        return cls(**data)

