from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed

from ..config import BacktestSettings
from .engine import run_backtest


def _compute_metrics(equity: pd.Series) -> tuple[float, float, float]:
    if equity.empty:
        return 0.0, 0.0, 0.0
    equity = equity.astype(float)
    end = float(equity.iloc[-1])
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, np.nan)
    max_dd = float(dd.max(skipna=True) or 0.0)
    # CAGR ~ roczna stopa z prostym przybliżeniem długości
    n = len(equity)
    years = max(1e-9, n / 252.0)  # trading-days approx
    cagr = (end / float(equity.iloc[0])) ** (1 / years) - 1 if end > 0 else -1.0
    return end, max_dd, cagr


def param_grid(fast_values: Iterable[int], slow_values: Iterable[int]) -> list[tuple[int, int]]:
    return [(f, s) for f, s in product(fast_values, slow_values) if f < s]


@dataclass
class GridResult:
    fast: int
    slow: int
    equity_end: float
    max_dd: float
    cagr: float


def run_grid(
    df: pd.DataFrame,
    symbol: str,
    fast_values: list[int],
    slow_values: list[int],
    capital: float = 100_000.0,
    risk: float = 0.01,
    max_dd: float = 0.30,
    fee: float = 0.0005,
    slippage: float = 0.0,
    atr_period: int = 14,
    atr_multiple: float = 2.0,
    use_rsi: bool = False,
    rsi_period: int = 14,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
    strategy_name: str = "ema_cross",
    signal: int = 9,
    time_model: Path | None = None,
    min_confluence: int = 1,
    blocked_hours: list[int] | None = None,
    blocked_weekdays: list[int] | None = None,
    n_jobs: int = 1,
    cache_dir: str = ".cache/forest5-grid",
) -> pd.DataFrame:
    blocked_hours = blocked_hours or []
    blocked_weekdays = blocked_weekdays or []

    mem = Memory(cache_dir, verbose=0)
    combos = param_grid(fast_values, slow_values)

    # Cache indicator computations to avoid recomputation for repeated parameters
    from ..core import indicators

    atr_orig, ema_orig = indicators.atr, indicators.ema
    indicators.atr = mem.cache(indicators.atr)
    indicators.ema = mem.cache(indicators.ema)

    @mem.cache
    def _single_run(fast: int, slow: int) -> GridResult:
        settings = BacktestSettings(
            symbol=symbol,
            timeframe="1h",
            strategy=dict(
                name=strategy_name,
                fast=fast,
                slow=slow,
                signal=signal,
                use_rsi=use_rsi,
                rsi_period=rsi_period,
                rsi_overbought=rsi_overbought,
                rsi_oversold=rsi_oversold,
            ),
            risk=dict(
                initial_capital=capital,
                risk_per_trade=risk,
                max_drawdown=max_dd,
                fee_perc=fee,
                slippage_perc=slippage,
            ),
            atr_period=atr_period,
            atr_multiple=atr_multiple,
        )
        settings.time.model.enabled = bool(time_model)
        settings.time.model.path = time_model
        settings.time.fusion_min_confluence = int(min_confluence)
        settings.time.blocked_hours = list(blocked_hours)
        settings.time.blocked_weekdays = list(blocked_weekdays)
        res = run_backtest(df, settings)
        end, mdd, cagr = _compute_metrics(res.equity_curve)
        return GridResult(fast=fast, slow=slow, equity_end=end, max_dd=mdd, cagr=cagr)

    try:
        if n_jobs == 1:
            results = [_single_run(f, s) for (f, s) in combos]
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(_single_run)(f, s) for (f, s) in combos)
    finally:
        indicators.atr = atr_orig
        indicators.ema = ema_orig

    out = pd.DataFrame([r.__dict__ for r in results])
    out["rar"] = out["cagr"] / out["max_dd"].replace(0, np.nan)
    out["rar"] = out["rar"].fillna(0.0)
    return out
