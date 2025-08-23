from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from .backtest.errors import BacktestConfigError
from .utils.timeframes import normalize_timeframe
from .config.strategy import BaseStrategySettings

# Default directory with historical CSV data. The path can be overridden via
# the ``FOREST5_DATA_DIR`` environment variable or an explicit configuration
# parameter passed to helper functions.
DEFAULT_DATA_DIR = Path("/home/daro/Fxdata")

# Commonly traded forex symbols. ``--symbol`` arguments in the CLI are
# restricted to this list.
ALLOWED_SYMBOLS = [
    "AUDUSD",
    "EURUSD",
    "EURJPY",
    "GBPJPY",
    "GBPUSD",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
]


def get_data_dir(override: str | Path | None = None) -> Path:
    """Return directory with OHLC CSV data.

    Priority order:
    ``override`` argument > ``FOREST5_DATA_DIR`` environment variable >
    :data:`DEFAULT_DATA_DIR`.
    """

    if override is not None:
        return Path(override)

    env = os.environ.get("FOREST5_DATA_DIR")
    if env:
        return Path(env)

    return DEFAULT_DATA_DIR


class StrategySettings(BaseStrategySettings):
    """Backtest strategy configuration."""

    pass


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
    fusion_min_confluence: float = 1.0

    @field_validator("q_high")
    @classmethod
    def _check_quantiles(cls, v: float, info):
        q_low = info.data.get("q_low")
        if not (0.0 <= q_low < v <= 1.0):
            raise BacktestConfigError("0.0 <= q_low < q_high <= 1.0")
        return v


class BacktestSettings(BaseModel):
    symbol: str = "SYMBOL"
    timeframe: str = "1h"
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    time: BacktestTimeSettings = Field(default_factory=BacktestTimeSettings)
    atr_period: int = 14
    atr_multiple: float = 2.0
    debug_dir: Path | None = None

    @field_validator("timeframe")
    @classmethod
    def _norm_tf(cls, v: str) -> str:
        return normalize_timeframe(v)

    @field_validator("debug_dir", mode="before")
    @classmethod
    def _to_path(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        return Path(v)

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
