from __future__ import annotations

import pandas as pd

from ..config_live import LiveSettings
from .factory import compute_signal
from .candles import candles_signal
from .combine import confirm_with_candles


def append_bar_and_signal(df: pd.DataFrame, bar: dict, settings: LiveSettings) -> int:
    """Append ``bar`` to ``df`` and return the latest trading signal.

    Supports incremental calculation for ``ema_cross`` and ``macd_cross``
    strategies. Falls back to :func:`compute_signal` for others.
    """
    idx = pd.to_datetime(bar["start"], unit="s")
    df.loc[idx, ["open", "high", "low", "close"]] = [
        bar["open"],
        bar["high"],
        bar["low"],
        bar["close"],
    ]

    name = getattr(settings.strategy, "name", "ema_cross")
    close = float(bar["close"])

    if name == "ema_cross":
        fast = settings.strategy.fast
        slow = settings.strategy.slow
        alpha_fast = 2 / (fast + 1)
        alpha_slow = 2 / (slow + 1)

        if "ema_fast" not in df.columns:
            df["ema_fast"] = pd.Series(dtype=float)
            df["ema_slow"] = pd.Series(dtype=float)

        if len(df) > 1:
            prev_fast = float(df["ema_fast"].iloc[-2])
            prev_slow = float(df["ema_slow"].iloc[-2])
        else:
            prev_fast = prev_slow = close

        ema_fast = close * alpha_fast + prev_fast * (1 - alpha_fast)
        ema_slow = close * alpha_slow + prev_slow * (1 - alpha_slow)
        df.at[idx, "ema_fast"] = ema_fast
        df.at[idx, "ema_slow"] = ema_slow

        sig = 0
        if len(df) > 1:
            prev_dir = 1 if prev_fast > prev_slow else -1
            direction = 1 if ema_fast > ema_slow else -1
            if direction != prev_dir:
                sig = direction
    elif name == "macd_cross":
        fast = settings.strategy.fast
        slow = settings.strategy.slow
        signal_period = getattr(settings.strategy, "signal", 9)
        a_fast = 2 / (fast + 1)
        a_slow = 2 / (slow + 1)
        a_sig = 2 / (signal_period + 1)

        if "ema_fast" not in df.columns:
            df["ema_fast"] = pd.Series(dtype=float)
            df["ema_slow"] = pd.Series(dtype=float)
            df["macd"] = pd.Series(dtype=float)
            df["macd_signal"] = pd.Series(dtype=float)

        if len(df) > 1:
            prev_fast = float(df["ema_fast"].iloc[-2])
            prev_slow = float(df["ema_slow"].iloc[-2])
            prev_macd = float(df["macd"].iloc[-2])
            prev_sig = float(df["macd_signal"].iloc[-2])
        else:
            prev_fast = prev_slow = close
            prev_macd = prev_fast - prev_slow
            prev_sig = prev_macd

        ema_fast = close * a_fast + prev_fast * (1 - a_fast)
        ema_slow = close * a_slow + prev_slow * (1 - a_slow)
        macd_val = ema_fast - ema_slow
        macd_sig = macd_val * a_sig + prev_sig * (1 - a_sig)

        df.at[idx, "ema_fast"] = ema_fast
        df.at[idx, "ema_slow"] = ema_slow
        df.at[idx, "macd"] = macd_val
        df.at[idx, "macd_signal"] = macd_sig

        sig = 0
        if len(df) > 1:
            prev_spread = prev_macd - prev_sig
            spread = macd_val - macd_sig
            prev_sgn = 1 if prev_spread > 0 else (-1 if prev_spread < 0 else 0)
            sgn = 1 if spread > 0 else (-1 if spread < 0 else 0)
            if sgn > prev_sgn and sgn > 0:
                sig = 1
            elif sgn < prev_sgn and sgn < 0:
                sig = -1
    else:
        return int(compute_signal(df, settings, "close").iloc[-1])

    candle = candles_signal(df.iloc[-2:]).iloc[-1] if len(df) > 1 else 0
    if sig != 0 or candle != 0:
        idx_ser = pd.Series([sig], index=[idx])
        candle_ser = pd.Series([candle], index=[idx])
        sig = int(confirm_with_candles(idx_ser, candle_ser).iloc[-1])
    return sig
