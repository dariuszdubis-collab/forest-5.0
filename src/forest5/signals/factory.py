from __future__ import annotations

import copy

from ..core.indicators import rsi
from .candles import candles_signal
from .combine import apply_rsi_filter, confirm_with_candles
from .ema import ema_cross_signal
from .macd import macd_cross_signal
from .h1_ema_rsi_atr import compute_primary_signal_h1
from ..utils.log import TelemetryContext


def compute_signal(
    df,
    settings,
    price_col="close",
    compat_int: bool = False,
    ctx: TelemetryContext | None = None,
):
    """Generate trading signal without mutating the input settings.

    ``settings`` may either be a complete settings object exposing a ``strategy``
    attribute or the strategy instance itself.  Only when ``compat_int`` is set
    to ``True`` will contract style signals be coerced to ``-1/0/1`` integers.
    """

    strategy_obj = getattr(settings, "strategy", settings)
    strategy = copy.deepcopy(strategy_obj)
    name = getattr(strategy, "name", "ema_cross")

    if name in {"ema_rsi", "ema-cross+rsi"}:
        name = "ema_cross"

    if name == "ema_cross":
        base = ema_cross_signal(df[price_col], strategy.fast, strategy.slow)
        if getattr(strategy, "use_rsi", False):
            rsi_series = rsi(df[price_col], strategy.rsi_period)
            base = apply_rsi_filter(
                base,
                rsi_series,
                strategy.rsi_overbought,
                strategy.rsi_oversold,
            )
        candles = candles_signal(df)
        return confirm_with_candles(base, candles)
    if name == "macd_cross":
        base = macd_cross_signal(
            df[price_col],
            strategy.fast,
            strategy.slow,
            getattr(strategy, "signal", 9),
        )
        if getattr(strategy, "use_rsi", False):
            rsi_series = rsi(df[price_col], strategy.rsi_period)
            base = apply_rsi_filter(
                base,
                rsi_series,
                strategy.rsi_overbought,
                strategy.rsi_oversold,
            )
        candles = candles_signal(df)
        return confirm_with_candles(base, candles)
    if name == "h1_ema_rsi_atr":
        params = getattr(strategy, "params", None)
        res = compute_primary_signal_h1(df, params, ctx=ctx)
        if compat_int:
            from .compat import contract_to_int

            return contract_to_int(res)
        return res
    raise ValueError(f"Unknown strategy: {name}")
