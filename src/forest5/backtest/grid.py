from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Callable, Any

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


def param_grid(
    params: Mapping[str, Iterable],
    filter_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return list of parameter combinations.

    Parameters are provided as a mapping from name to an iterable of values.
    Optionally a ``filter_fn`` can be supplied to drop unwanted combinations.
    """

    keys = list(params.keys())
    combos = [dict(zip(keys, values)) for values in product(*params.values())]
    if filter_fn:
        combos = [c for c in combos if filter_fn(c)]
    return combos


def run_grid(
    df: pd.DataFrame,
    symbol: str,
    fast_values: Iterable[int],
    slow_values: Iterable[int],
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
    time_model: Path | None = None,
    min_confluence: int = 1,
    blocked_hours: list[int] | None = None,
    blocked_weekdays: list[int] | None = None,
    n_jobs: int = 1,
    cache_dir: str = ".cache/forest5-grid",
    **param_lists: Iterable,
) -> pd.DataFrame:
    blocked_hours = blocked_hours or []
    blocked_weekdays = blocked_weekdays or []

    mem = Memory(cache_dir, verbose=0)
    param_values: dict[str, Iterable] = {"fast": fast_values, "slow": slow_values}
    for name, values in param_lists.items():
        if name.endswith("_values"):
            name = name[:-7]
        param_values[name] = values
    combos = param_grid(param_values, lambda p: p["fast"] < p["slow"])

    # Cache indicator computations to avoid recomputation for repeated parameters
    from ..core import indicators

    atr_orig, ema_orig = indicators.atr, indicators.ema
    indicators.atr = mem.cache(indicators.atr)
    indicators.ema = mem.cache(indicators.ema)

    @mem.cache
    def _single_run(**params: Any) -> dict[str, Any]:
        fast = params["fast"]
        slow = params["slow"]
        settings = BacktestSettings(
            symbol=symbol,
            timeframe="1h",
            strategy=dict(
                name="ema_cross",
                fast=fast,
                slow=slow,
                use_rsi=params.get("use_rsi", use_rsi),
                rsi_period=params.get("rsi_period", rsi_period),
                rsi_overbought=params.get("rsi_overbought", rsi_overbought),
                rsi_oversold=params.get("rsi_oversold", rsi_oversold),
            ),
            risk=dict(
                initial_capital=params.get("capital", capital),
                risk_per_trade=params.get("risk", risk),
                max_drawdown=params.get("max_dd", max_dd),
                fee_perc=params.get("fee", fee),
                slippage_perc=params.get("slippage", slippage),
            ),
            atr_period=params.get("atr_period", atr_period),
            atr_multiple=params.get("atr_multiple", atr_multiple),
        )
        tm = params.get("time_model", time_model)
        settings.time.model.enabled = bool(tm)
        settings.time.model.path = tm
        settings.time.fusion_min_confluence = int(params.get("min_confluence", min_confluence))
        settings.time.blocked_hours = list(params.get("blocked_hours", blocked_hours))
        settings.time.blocked_weekdays = list(params.get("blocked_weekdays", blocked_weekdays))
        res = run_backtest(df, settings)
        end, mdd, cagr = _compute_metrics(res.equity_curve)
        out = dict(params)
        out.update({"equity_end": end, "max_dd": mdd, "cagr": cagr})
        return out

    try:
        if n_jobs == 1:
            results = [_single_run(**p) for p in combos]
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(_single_run)(**p) for p in combos)
    finally:
        indicators.atr = atr_orig
        indicators.ema = ema_orig

    out = pd.DataFrame(results)
    out["rar"] = out["cagr"] / out["max_dd"].replace(0, np.nan)
    out["rar"] = out["rar"].fillna(0.0)
    return out
