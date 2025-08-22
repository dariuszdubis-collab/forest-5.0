from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Any, Dict, List

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


def param_grid(param_lists: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """Generate a grid of parameters for any number of dimensions.

    Parameters
    ----------
    param_lists: dict
        Mapping from parameter name to an iterable of possible values.

    Returns
    -------
    list of dict
        Each dict contains one concrete combination of parameters.
    """

    keys = list(param_lists.keys())
    return [dict(zip(keys, values)) for values in product(*[param_lists[k] for k in keys])]


@dataclass
class GridResult:
    fast: int
    slow: int
    risk: float
    rsi_period: int
    max_dd: float
    equity_end: float
    dd: float
    cagr: float


def run_grid(
    df: pd.DataFrame,
    symbol: str,
    fast_values: list[int],
    slow_values: list[int],
    risk_values: list[float] | None = None,
    rsi_period_values: list[int] | None = None,
    max_dd_values: list[float] | None = None,
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
    min_confluence: float = 1.0,
    n_jobs: int = 1,
    cache_dir: str = ".cache/forest5-grid",
    debug_dir: Path | None = None,
) -> pd.DataFrame:

    mem = Memory(cache_dir, verbose=0)
    param_lists = {
        "fast": fast_values,
        "slow": slow_values,
    }
    if risk_values is not None:
        param_lists["risk"] = risk_values
    if rsi_period_values is not None:
        param_lists["rsi_period"] = rsi_period_values
    if max_dd_values is not None:
        param_lists["max_dd"] = max_dd_values

    combos = [c for c in param_grid(param_lists) if c["fast"] < c["slow"]]

    # Cache indicator computations to avoid recomputation for repeated parameters
    from ..core import indicators

    atr_orig, ema_orig = indicators.atr, indicators.ema
    indicators.atr = mem.cache(indicators.atr)
    indicators.ema = mem.cache(indicators.ema)

    base_debug_dir = Path(debug_dir) if debug_dir else None

    @mem.cache
    def _single_run(
        fast: int,
        slow: int,
        risk_value: float,
        rsi_period_value: int,
        max_dd_value: float,
    ) -> GridResult:
        run_debug = None
        if base_debug_dir:
            name = (
                f"fast{fast}_slow{slow}_risk{risk_value}_"
                f"rsi{rsi_period_value}_maxdd{max_dd_value}"
            )
            run_debug = base_debug_dir / name
        settings = BacktestSettings(
            symbol=symbol,
            timeframe="1h",
            strategy=dict(
                name="ema_cross",
                fast=fast,
                slow=slow,
                use_rsi=use_rsi,
                rsi_period=rsi_period_value,
                rsi_overbought=rsi_overbought,
                rsi_oversold=rsi_oversold,
            ),
            risk=dict(
                initial_capital=capital,
                risk_per_trade=risk_value,
                max_drawdown=max_dd_value,
                fee_perc=fee,
                slippage_perc=slippage,
            ),
            atr_period=atr_period,
            atr_multiple=atr_multiple,
            debug_dir=run_debug,
        )
        settings.time.model.enabled = bool(time_model)
        settings.time.model.path = time_model
        settings.time.fusion_min_confluence = float(min_confluence)
        res = run_backtest(df, settings)
        end, mdd, cagr = _compute_metrics(res.equity_curve)
        return GridResult(
            fast=fast,
            slow=slow,
            risk=risk_value,
            rsi_period=rsi_period_value,
            max_dd=max_dd_value,
            equity_end=end,
            dd=mdd,
            cagr=cagr,
        )

    try:
        if n_jobs == 1:
            results = [
                _single_run(
                    c["fast"],
                    c["slow"],
                    c.get("risk", risk),
                    c.get("rsi_period", rsi_period),
                    c.get("max_dd", max_dd),
                )
                for c in combos
            ]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_single_run)(
                    c["fast"],
                    c["slow"],
                    c.get("risk", risk),
                    c.get("rsi_period", rsi_period),
                    c.get("max_dd", max_dd),
                )
                for c in combos
            )
    finally:
        indicators.atr = atr_orig
        indicators.ema = ema_orig

    out = pd.DataFrame([r.__dict__ for r in results])
    out["rar"] = out["cagr"] / out["dd"].replace(0, np.nan)
    out["rar"] = out["rar"].fillna(0.0)
    return out
