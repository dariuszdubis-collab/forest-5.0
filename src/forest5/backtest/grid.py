from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Memory

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
    atr_period: int = 14,
    atr_multiple: float = 2.0,
    n_jobs: int = 1,
    cache_dir: str = ".cache/forest5-grid",
) -> pd.DataFrame:
    mem = Memory(cache_dir, verbose=0)
    combos = param_grid(fast_values, slow_values)

    @mem.cache
    def _single_run(fast: int, slow: int) -> GridResult:
        settings = BacktestSettings(
            symbol=symbol,
            timeframe="1h",
            strategy=dict(name="ema_cross", fast=fast, slow=slow),
            risk=dict(initial_capital=capital, risk_per_trade=risk, max_drawdown=max_dd),
            atr_period=atr_period,
            atr_multiple=atr_multiple,
        )
        res = run_backtest(df, settings)
        end, mdd, cagr = _compute_metrics(res.equity_curve)
        return GridResult(fast=fast, slow=slow, equity_end=end, max_dd=mdd, cagr=cagr)

    results = (
        [_single_run(f, s) for (f, s) in combos]
        if n_jobs == 1
        else mem.cache(lambda c: [_single_run(f, s) for (f, s) in c])(combos)
    )

    out = pd.DataFrame([r.__dict__ for r in results])
    out["rar"] = out["cagr"] / out["max_dd"].replace(0, np.nan)
    out["rar"] = out["rar"].fillna(0.0)
    return out
