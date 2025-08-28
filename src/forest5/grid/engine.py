from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Any, Callable
import random
from copy import deepcopy

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from ..config import BacktestSettings
from ..backtest.engine import run_backtest
from ..backtest.grid import _compute_metrics, build_combo_id
from ..core.indicators import precompute_indicators
from ..utils.log import log_event


def plan_param_grid(
    param_ranges: Dict[str, Iterable[Any]],
    *,
    filter_fn: Callable[[Dict[str, Any]], bool] | None = None,
) -> pd.DataFrame:
    """Enumerate parameter combinations and assign a ``combo_id``.

    Parameters
    ----------
    param_ranges:
        Mapping from parameter name to an iterable of possible values.
    filter_fn:
        Optional predicate to discard combinations.  It receives a dict of
        parameters and should return ``True`` to keep the combination.

    Returns
    -------
    pandas.DataFrame
        DataFrame where each row represents a parameter combination and the
        first column ``combo_id`` is a unique identifier.
    """

    keys = sorted(param_ranges.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[param_ranges[k] for k in keys])]
    if filter_fn is not None:
        combos = [c for c in combos if filter_fn(c)]
    df = pd.DataFrame(combos)
    df.insert(0, "combo_id", [build_combo_id(c) for c in combos])
    return df


def run_grid(
    df: pd.DataFrame,
    combos: pd.DataFrame,
    base_settings: BacktestSettings,
    *,
    jobs: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Execute backtests for prepared parameter combinations."""

    combo_dicts = combos.to_dict("records")

    # Precompute indicators used across the grid
    set_fast = set(combos["ema_fast"]) if "ema_fast" in combos else set()
    set_slow = set(combos["ema_slow"]) if "ema_slow" in combos else set()
    set_rsi = set(combos["rsi_period"]) if "rsi_period" in combos else set()
    set_atr = set(combos["atr_period"]) if "atr_period" in combos else set()
    created_cols = precompute_indicators(
        df,
        ema_periods=set_fast | set_slow,
        rsi_periods=set_rsi,
        atr_periods=set_atr,
    )
    log_event("grid.indicators.cached", count=len(created_cols))

    def _single(idx: int, combo: Dict[str, Any]) -> Dict[str, Any]:
        if seed is not None:
            local_seed = seed + idx
            random.seed(local_seed)
            np.random.seed(local_seed)

        settings = (
            base_settings.model_copy(deep=True)
            if hasattr(base_settings, "model_copy")
            else deepcopy(base_settings)
        )
        for k, v in combo.items():
            if k == "combo_id":
                continue
            if hasattr(settings.strategy, k):
                setattr(settings.strategy, k, v)
            elif hasattr(settings.risk, k):
                setattr(settings.risk, k, v)
            elif hasattr(settings.time, k):
                setattr(settings.time, k, v)
            else:
                setattr(settings, k, v)

        try:
            res = run_backtest(df, settings)
            equity = res.equity_curve
            end, dd, cagr = _compute_metrics(equity)
            tb = res.trades.trades
            pnls = [t.pnl for t in tb]
            pnl = float(sum(pnls))
            pnl_net = end - float(settings.risk.initial_capital)
            trades = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            winrate = wins / trades if trades else 0.0
            expectancy = pnl / trades if trades else 0.0
            returns = equity.pct_change().dropna()
            sharpe = (
                float(returns.mean() / returns.std() * np.sqrt(252))
                if not returns.empty and returns.std() != 0
                else 0.0
            )
            return {
                **combo,
                "equity_end": end,
                "dd": dd,
                "cagr": cagr,
                "rar": cagr / dd if dd else 0.0,
                "trades": trades,
                "winrate": winrate,
                "pnl": pnl,
                "pnl_net": pnl_net,
                "sharpe": sharpe,
                "expectancy": expectancy,
                "timeonly_wait_pct": 0.0,
                "setups_expired_pct": 0.0,
            }
        except Exception as exc:  # pragma: no cover
            return {**combo, "error": str(exc)}

    if jobs == 1:
        rows = [_single(i, c) for i, c in enumerate(combo_dicts)]
    else:
        rows = Parallel(n_jobs=jobs)(delayed(_single)(i, c) for i, c in enumerate(combo_dicts))

    return pd.DataFrame(rows)
