from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from .backtest.errors import BacktestConfigError
from .utils.timeframes import normalize_timeframe


class StrategySettings(BaseModel):
    name: Literal["ema_cross", "macd_cross"] = "ema_cross"
    fast: int = 12
    slow: int = 26
    signal: int = 9
    use_rsi: bool = False
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30


class OnDrawdownSettings(BaseModel):
    action: Literal["halt", "soft_wait"] = "halt"


class RiskSettings(BaseModel):
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    max_drawdown: float = 0.30
    fee_perc: float = 0.0005
    slippage_perc: float = 0.0
    on_drawdown: OnDrawdownSettings = Field(default_factory=OnDrawdownSettings)


class AISettings(BaseModel):
    enabled: bool = False
    model: str = "gpt-4o-mini"
    max_tokens: int = 256
    context_file: str | None = None


class TimeOnlySettings(BaseModel):
    enabled: bool = False
    blocked_weekdays: list[int] = Field(default_factory=list)  # 0=Mon..6=Sun
    blocked_hours: list[int] = Field(default_factory=list)  # 0..23


class BacktestTimeModelSettings(BaseModel):
    enabled: bool = False
    path: Path | None = None

    @field_validator("path", mode="before")
    @classmethod
    def _to_path(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        return Path(v)


class BacktestTimeSettings(BaseModel):
    model: BacktestTimeModelSettings = Field(default_factory=BacktestTimeModelSettings)
    q_low: float = 0.1
    q_high: float = 0.9
    blocked_hours: list[int] = Field(default_factory=list)
    blocked_weekdays: list[int] = Field(default_factory=list)
    fusion_min_confluence: int = 1

    @field_validator("q_high")
    @classmethod
    def _check_quantiles(cls, v: float, info):
        q_low = info.data.get("q_low")
        if not (0.0 <= q_low < v <= 1.0):
            raise BacktestConfigError("0.0 <= q_low < q_high <= 1.0")
        return v

    @field_validator("blocked_weekdays")
    @classmethod
    def _check_weekdays(cls, v: list[int]) -> list[int]:
        invalid = [d for d in v if d < 0 or d > 6]
        if invalid:
            raise BacktestConfigError(f"blocked_weekdays must be in range 0-6: {invalid}")
        return v

    @field_validator("blocked_hours")
    @classmethod
    def _check_hours(cls, v: list[int]) -> list[int]:
        invalid = [h for h in v if h < 0 or h > 23]
        if invalid:
            raise BacktestConfigError(f"blocked_hours must be in range 0-23: {invalid}")
        return v


class BacktestSettings(BaseModel):
    symbol: str = "SYMBOL"
    timeframe: str = "1h"
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    time: BacktestTimeSettings = Field(default_factory=BacktestTimeSettings)
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
            raise BacktestConfigError("Supported: .yaml/.yml/.json")
        return cls(**data)
