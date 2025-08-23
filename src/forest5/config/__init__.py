from __future__ import annotations

# Backwards compatibility: re-export settings from the legacy config.py
import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "forest5._legacy_config", Path(__file__).resolve().parent.parent / "config.py"
)
_legacy = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None  # nosec B101
sys.modules[_spec.name] = _legacy  # register before execution for pydantic
_spec.loader.exec_module(_legacy)  # type: ignore[assignment]

# Ensure pydantic models are fully built
for _model in (
    _legacy.StrategySettings,
    _legacy.RiskSettings,
    _legacy.OnDrawdownSettings,
    _legacy.AISettings,
    _legacy.TimeOnlySettings,
    _legacy.BacktestTimeModelSettings,
    _legacy.BacktestTimeSettings,
    _legacy.BacktestSettings,
):
    if hasattr(_model, "model_rebuild"):
        _model.model_rebuild()

# Re-export models and adjust __module__ for pickling
for _name in [
    "StrategySettings",
    "RiskSettings",
    "OnDrawdownSettings",
    "AISettings",
    "TimeOnlySettings",
    "BacktestTimeModelSettings",
    "BacktestTimeSettings",
    "BacktestSettings",
]:
    _cls = getattr(_legacy, _name)
    _cls.__module__ = __name__
    globals()[_name] = _cls

DEFAULT_DATA_DIR = _legacy.DEFAULT_DATA_DIR
ALLOWED_SYMBOLS = _legacy.ALLOWED_SYMBOLS
get_data_dir = _legacy.get_data_dir

from .loader import load_live_settings  # noqa: E402

__all__ = [
    "load_live_settings",
    "StrategySettings",
    "RiskSettings",
    "OnDrawdownSettings",
    "AISettings",
    "TimeOnlySettings",
    "BacktestTimeModelSettings",
    "BacktestTimeSettings",
    "BacktestSettings",
    "DEFAULT_DATA_DIR",
    "ALLOWED_SYMBOLS",
    "get_data_dir",
]
