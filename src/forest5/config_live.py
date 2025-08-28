from __future__ import annotations

from pathlib import Path
import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import (
    ALLOWED_SYMBOLS,
    RiskSettings as BaseRiskSettings,
    AISettings as BaseAISettings,
)
from .config.strategy import BaseStrategySettings, PatternSettings
from .utils.timeframes import normalize_timeframe


class StrategySettings(BaseStrategySettings):
    timeframe: str = "1m"

    @field_validator("timeframe")
    @classmethod
    def _norm_tf(cls, v: str) -> str:
        return normalize_timeframe(v)


class RiskSettings(BaseRiskSettings):
    @field_validator("risk_per_trade")
    @classmethod
    def _check_rpt(cls, v: float) -> float:
        if not (0 < v <= 0.05):
            raise ValueError("risk.risk_per_trade must be in (0, 0.05]")
        return v

    @field_validator("max_drawdown")
    @classmethod
    def _check_max_dd(cls, v: float) -> float:
        if not (0 < v <= 1.0):
            raise ValueError("risk.max_drawdown must be in (0, 1]")
        return v


class AISettings(BaseAISettings):
    @model_validator(mode="after")
    def _check_context(self) -> "AISettings":
        if self.enabled and not (self.context_file and Path(self.context_file).exists()):
            raise ValueError("ai.context_file missing")
        return self


class BrokerSettings(BaseModel):
    type: Literal["mt4", "paper"]
    bridge_dir: Path | None = None
    symbol: str
    volume: float = 1.0
    timeframe: str = "1m"
    stop_level_points: float | None = None

    @field_validator("bridge_dir", mode="before")
    @classmethod
    def _to_path(cls, v: str | Path | None) -> Path | None:
        if v in (None, ""):
            return None
        return Path(v)

    @field_validator("symbol")
    @classmethod
    def _check_symbol(cls, v: str) -> str:
        if v not in ALLOWED_SYMBOLS:
            raise ValueError("unsupported broker.symbol")
        return v

    @field_validator("volume")
    @classmethod
    def _check_volume(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("volume must be > 0")
        return v

    @model_validator(mode="after")
    def _check_stop_level(self) -> "BrokerSettings":
        if self.type == "mt4" and self.stop_level_points is not None and self.stop_level_points < 0:
            raise ValueError("stop_level_points must be >= 0")
        return self


class DecisionTechSettings(BaseModel):
    default_conf_int: float = 0.50
    conf_floor: float = 0.20
    conf_cap: float = 0.90


class DecisionWeights(BaseModel):
    tech: float = 1.0
    ai: float = 0.5


class DecisionSettings(BaseModel):
    min_confluence: float = 0.0
    tie_epsilon: float = 0.05
    weights: DecisionWeights = Field(default_factory=DecisionWeights)
    tech: DecisionTechSettings = Field(default_factory=DecisionTechSettings)

    @field_validator("min_confluence")
    @classmethod
    def _check_confluence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("0.0 <= decision.min_confluence <= 1.0")
        return v


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


class PrimarySignalSettings(BaseModel):
    strategy: BaseStrategySettings = Field(default_factory=BaseStrategySettings)
    patterns: PatternSettings = Field(default_factory=PatternSettings)


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
    "DecisionTechSettings",
    "DecisionWeights",
    "DecisionSettings",
    "LiveTimeModelSettings",
    "PrimarySignalSettings",
    "LiveTimeSettings",
    "LiveSettings",
    "validate_live_config",
]


def validate_live_config(path: str | Path, strict: bool = False) -> tuple[bool, dict]:
    """Validate a live trading configuration file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    strict:
        Currently unused placeholder for future extensions.

    Returns
    -------
    tuple[bool, dict]
        ``(True, details)`` on success, ``(False, details)`` on failure.  The
        ``details`` dictionary contains either a ``message`` or ``error`` field.
    """

    from .config.loader import load_live_settings

    try:
        settings = load_live_settings(path)
    except Exception as exc:  # pragma: no cover - defensive
        return False, {"error": str(exc)}

    broker = settings.broker
    if not broker.bridge_dir:
        return False, {"error": "Missing fields: broker.bridge_dir"}

    bridge_dir = Path(broker.bridge_dir)
    spec_path = bridge_dir / "symbol_specs.json"
    if spec_path.exists():
        try:
            specs = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            return False, {"error": f"Invalid symbol_specs.json: {exc}"}
        required = {"digits", "point"}
        if "stop_level" not in specs and "STOPLEVEL" not in specs:
            required.add("stop_level")
        if not required.issubset(specs):
            return False, {"error": "symbol_specs.json missing required fields"}

    message = f"OK: {broker.symbol} @ {broker.type}"
    return True, {"message": message}
