from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class BaseStrategySettings(BaseModel):
    """Common strategy configuration shared between backtest and live settings."""

    name: Literal["ema_cross", "macd_cross"] = "ema_cross"
    fast: int = 12
    slow: int = 26
    signal: int = 9
    use_rsi: bool = False
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30


__all__ = ["BaseStrategySettings"]
