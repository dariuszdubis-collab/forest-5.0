from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class H1EmaRsiAtrParams(BaseModel):
    """Parameter container for the ``h1_ema_rsi_atr`` strategy."""

    ema_fast: int = 21
    ema_slow: int = 55
    atr_period: int = 14
    rsi_period: int = 14


class H1EmaRsiAtrSettings(BaseModel):
    """Settings for the ``h1_ema_rsi_atr`` strategy."""

    name: Literal["h1_ema_rsi_atr"] = "h1_ema_rsi_atr"
    compat_int: int | None = None
    params: H1EmaRsiAtrParams = Field(default_factory=H1EmaRsiAtrParams)


class BaseStrategySettings(BaseModel):
    """Common strategy configuration shared between backtest and live settings."""

    name: Literal["ema_cross", "macd_cross", "h1_ema_rsi_atr"] = "ema_cross"
    fast: int = 12
    slow: int = 26
    signal: int = 9
    use_rsi: bool = False
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    compat_int: int | None = None
    params: dict[str, Any] | None = None


__all__ = ["BaseStrategySettings", "H1EmaRsiAtrSettings"]
