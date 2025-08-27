from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, model_validator


class H1EmaRsiAtrParams(BaseModel):
    """Parameter container for the ``h1_ema_rsi_atr`` strategy."""

    model_config = ConfigDict(populate_by_name=True)

    ema_fast: int = 21
    ema_slow: int = 55
    atr_period: int = 14
    rsi_period: int = 14
    t_sep_atr: float = 0.5
    pullback_atr: float = Field(0.5, alias="pullback_to_ema_fast_atr")
    entry_buffer_atr: float = 0.1
    sl_buffer_atr: float = 0.0
    tp_buffer_atr: float = 0.0
    sl_atr: float = 1.0
    sl_min_atr: float = 0.0
    rr: float = 2.0
    timeframe: str = "H1"
    horizon_minutes: int = 240

    def __getitem__(self, item: str) -> Any:  # pragma: no cover - convenience
        return getattr(self, item)


class PatternSettings(BaseModel):
    enabled: bool = True
    min_strength: float = 0.6
    boost_conf: float = 0.20
    boost_score: float = 0.20
    gate: bool = False
    engulfing: bool = True
    pinbar: bool = True
    stars: bool = True


class ProfileSettings(BaseModel):
    horizon_minutes: int = 240


class TimeModelQuantiles(BaseModel):
    q_low: float = 0.1
    q_high: float = 0.9


class H1EmaRsiAtrSettings(BaseModel):
    """Settings for the ``h1_ema_rsi_atr`` strategy."""

    name: Literal["h1_ema_rsi_atr"] = "h1_ema_rsi_atr"
    compat_int: int | None = None
    params: H1EmaRsiAtrParams = Field(default_factory=H1EmaRsiAtrParams)
    patterns: PatternSettings = Field(default_factory=PatternSettings)
    profile: ProfileSettings = Field(default_factory=ProfileSettings)
    time: TimeModelQuantiles = Field(default_factory=TimeModelQuantiles)
    tp_sl_priority: Literal["SL_FIRST", "TP_FIRST"] = "SL_FIRST"
    setup_ttl_bars: int = 1
    setup_ttl_minutes: int | None = None


class BaseStrategySettings(BaseModel):
    """Common strategy configuration shared between backtest and live settings."""

    model_config = ConfigDict(populate_by_name=True)

    name: Literal["ema_cross", "macd_cross", "h1_ema_rsi_atr"] = "ema_cross"
    fast: int = 12
    slow: int = 26
    signal: int = 9
    use_rsi: bool = False
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    compat_int: int | None = None
    params: H1EmaRsiAtrParams | dict[str, Any] | None = None
    patterns: PatternSettings = Field(default_factory=PatternSettings)
    tp_sl_priority: Literal["SL_FIRST", "TP_FIRST"] = "SL_FIRST"
    setup_ttl_bars: int = 1
    setup_ttl_minutes: int | None = None

    # Optional overrides for ``h1_ema_rsi_atr`` parameters from CLI/ENV
    ema_fast: int | None = None
    ema_slow: int | None = None
    atr_period: int | None = None
    t_sep_atr: float | None = None
    pullback_atr: float | None = Field(None, alias="pullback_to_ema_fast_atr")
    entry_buffer_atr: float | None = None
    sl_buffer_atr: float | None = None
    tp_buffer_atr: float | None = None
    sl_atr: float | None = None
    sl_min_atr: float | None = None
    rr: float | None = None
    timeframe: str | None = None
    horizon_minutes: int | None = None

    @model_validator(mode="after")
    def _map_h1_params(self) -> "BaseStrategySettings":
        if self.name != "h1_ema_rsi_atr":
            return self
        data: dict[str, Any]
        if isinstance(self.params, H1EmaRsiAtrParams):
            data = self.params.model_dump(by_alias=False)
        else:
            data = dict(self.params or {})
        for field in (
            "ema_fast",
            "ema_slow",
            "atr_period",
            "rsi_period",
            "t_sep_atr",
            "pullback_atr",
            "entry_buffer_atr",
            "sl_buffer_atr",
            "tp_buffer_atr",
            "sl_atr",
            "sl_min_atr",
            "rr",
            "timeframe",
            "horizon_minutes",
        ):
            value = getattr(self, field, None)
            if value is not None:
                data[field] = value
        self.params = H1EmaRsiAtrParams(**data)
        return self


__all__ = ["BaseStrategySettings", "H1EmaRsiAtrSettings", "PatternSettings"]
