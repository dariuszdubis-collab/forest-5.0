from __future__ import annotations

from pathlib import Path
import json

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import RiskSettings, AISettings
from .config.strategy import BaseStrategySettings, PatternSettings
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

    missing: list[str] = []
    broker = getattr(settings, "broker", None)
    btype = getattr(broker, "type", "") if broker else ""
    if not btype:
        missing.append("broker.type")
    else:
        allowed = {"mt4", "mt4_stub", "file", "paper"}
        if btype not in allowed:
            return False, {"error": "Invalid broker.type"}
    if broker is None or not getattr(broker, "bridge_dir", None):
        missing.append("broker.bridge_dir")
    if broker is None or not getattr(broker, "symbol", "").strip():
        missing.append("broker.symbol")

    risk = getattr(settings, "risk", None)
    for field in ("initial_capital", "risk_per_trade", "max_drawdown"):
        if risk is None or getattr(risk, field, None) in (None, ""):
            missing.append(f"risk.{field}")

    ai = getattr(settings, "ai", None)
    if ai and getattr(ai, "enabled", False):
        ctx_file = getattr(ai, "context_file", "") or ""
        require_ctx = getattr(ai, "require_context", False)
        if (not ctx_file) and require_ctx:
            missing.append("ai.context_file")

    if missing:
        return False, {"error": f"Missing fields: {', '.join(missing)}"}

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
