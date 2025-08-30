from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Any, Dict, List, Mapping
import json
import hashlib
import random
from copy import deepcopy
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn

import math
import warnings

from ..config import BacktestSettings
from .engine import run_backtest
from ..utils.io import atomic_to_csv, atomic_write_json

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def make_combo_id(params: Mapping[str, Any]) -> str:
    """Return a deterministic id for a parameter set."""

    s = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def build_combo_id(params: Mapping[str, Any]) -> str:  # pragma: no cover - backwards compat
    """Backward compatible wrapper for ``make_combo_id``."""

    return make_combo_id(params)


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
    t_sep_atr: float = 0.5,
    pullback_atr: float = 0.5,
    entry_buffer_atr: float = 0.1,
    sl_atr: float = 1.0,
    sl_min_atr: float = 0.0,
    rr: float = 2.0,
    q_low: float = 0.1,
    q_high: float = 0.9,
    strategy: str = "ema_cross",
    patterns: Dict[str, Dict[str, bool]] | None = None,
    time_model: Path | None = None,
    min_confluence: float = 1.0,
    n_jobs: int = 1,
    cache_dir: str = ".cache/forest5-grid",
    debug_dir: Path | None = None,
    seed: int | None = None,
    setup_ttl_minutes: int | None = None,
    resume: bool | str = "auto",
    chunks: int = 1,
    chunk_id: int | None = None,
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
        seed_value: int | None = None,
    ) -> GridResult:
        if seed_value is not None:
            random.seed(seed_value)
            np.random.seed(seed_value)
        run_debug = None
        if base_debug_dir:
            name = (
                f"fast{fast}_slow{slow}_risk{risk_value}_"
                f"rsi{rsi_period_value}_maxdd{max_dd_value}"
            )
            run_debug = base_debug_dir / name
        if strategy == "h1_ema_rsi_atr":
            strat = dict(
                name=strategy,
                ema_fast=fast,
                ema_slow=slow,
                atr_period=atr_period,
                rsi_period=rsi_period_value,
                t_sep_atr=t_sep_atr,
                pullback_atr=pullback_atr,
                entry_buffer_atr=entry_buffer_atr,
                sl_atr=sl_atr,
                sl_min_atr=sl_min_atr,
                rr=rr,
            )
            if patterns:
                strat["patterns"] = patterns
        else:
            strat = dict(
                name=strategy,
                fast=fast,
                slow=slow,
                use_rsi=use_rsi,
                rsi_period=rsi_period_value,
                rsi_overbought=rsi_overbought,
                rsi_oversold=rsi_oversold,
            )
        settings = BacktestSettings(
            symbol=symbol,
            timeframe="1h",
            strategy=strat,
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
            setup_ttl_minutes=setup_ttl_minutes,
        )
        settings.time.model.enabled = bool(time_model)
        settings.time.model.path = time_model
        settings.time.fusion_min_confluence = float(min_confluence)
        settings.time.q_low = float(q_low)
        settings.time.q_high = float(q_high)
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
            results = []
            for i, c in enumerate(combos):
                seed_value = seed + i if seed is not None else None
                results.append(
                    _single_run(
                        c["fast"],
                        c["slow"],
                        c.get("risk", risk),
                        c.get("rsi_period", rsi_period),
                        c.get("max_dd", max_dd),
                        seed_value,
                    )
                )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_single_run)(
                    c["fast"],
                    c["slow"],
                    c.get("risk", risk),
                    c.get("rsi_period", rsi_period),
                    c.get("max_dd", max_dd),
                    seed + i if seed is not None else None,
                )
                for i, c in enumerate(combos)
            )
    finally:
        indicators.atr = atr_orig
        indicators.ema = ema_orig

    out = pd.DataFrame([r.__dict__ for r in results])
    out.loc[:, "rar"] = out["cagr"] / out["dd"].replace(0, np.nan)
    out.loc[:, "rar"] = out["rar"].fillna(0.0)
    return out


def run_param_grid(
    df: pd.DataFrame,
    base_settings: BacktestSettings,
    param_ranges: Dict[str, Iterable[Any]],
    *,
    jobs: int = 0,
    seed: int | None = None,
    dry_run: bool = False,
    results_path: Path | None = None,
    meta_path: Path | None = None,
):
    """Run a grid search over arbitrary parameter ranges.

    Parameters
    ----------
    df:
        OHLC data.
    base_settings:
        :class:`BacktestSettings` instance providing base configuration.
    param_ranges:
        Mapping of parameter names to iterables of values.  Keys matching
        attributes on ``settings.strategy``/``settings.risk``/``settings.time``
        are applied accordingly.
    jobs:
        Number of parallel workers.
    seed:
        Optional base seed.  Each worker receives ``seed + index``.
    dry_run:
        If ``True`` only combinations are returned without running backtests.
    results_path, meta_path:
        Optional paths where ``results.csv`` and ``meta.json`` should be
        written.
    """

    df = df.copy()
    keys = sorted(param_ranges.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[param_ranges[k] for k in keys])]
    from datetime import datetime, timezone
    import subprocess  # nosec B404

    git_rev = "unknown"
    try:  # pragma: no cover - best effort
        git_rev = (
            subprocess.check_output(  # nosec
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover
        pass  # nosec B110

    meta = {
        "params": param_ranges,
        "n_combos": len(combos),
        "seed": seed,
        "git": git_rev,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Optional: cache indicators globally to save recomputation across combos
    from ..core import indicators as _ind
    from joblib import Memory as _Memory
    _mem = _Memory(".cache/forest5-grid", verbose=0)
    _atr_orig, _ema_orig = _ind.atr, _ind.ema
    _ind.atr = _mem.cache(_ind.atr)
    _ind.ema = _mem.cache(_ind.ema)

    if dry_run:
        out = pd.DataFrame(combos)
        out.insert(
            0, "combo_json", [json.dumps(c, sort_keys=True, separators=(",", ":")) for c in combos]
        )
        out.insert(0, "combo_id", [make_combo_id(c) for c in combos])
        if results_path:
            atomic_to_csv(out, results_path)
        if meta_path:
            atomic_write_json(meta, meta_path)
        return out, meta

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
            if hasattr(settings.strategy, k):
                setattr(settings.strategy, k, v)
            elif hasattr(settings.risk, k):
                setattr(settings.risk, k, v)
            elif hasattr(settings.time, k):
                setattr(settings.time, k, v)
            else:
                setattr(settings, k, v)
        combo_json = json.dumps(combo, sort_keys=True, separators=(",", ":"))
        combo_id = make_combo_id(combo)
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
            pattern_map: Dict[str, List[float]] = {}
            rr_vals: List[float] = []
            for t in tb:
                if t.pattern:
                    pattern_map.setdefault(t.pattern, []).append(t.pnl)
                if t.sl is not None and t.entry is not None and t.qty:
                    risk_amt = abs((t.entry - t.sl) * t.qty)
                    if risk_amt > 0:
                        rr_vals.append(t.pnl / risk_amt)
            expectancy_by_pattern = {k: float(np.mean(v)) for k, v in pattern_map.items()}
            rr_avg = float(np.mean(rr_vals)) if rr_vals else 0.0
            rr_median = float(np.median(rr_vals)) if rr_vals else 0.0
            returns = equity.pct_change().dropna()
            sharpe = (
                float(returns.mean() / returns.std() * np.sqrt(252))
                if not returns.empty and returns.std() != 0
                else 0.0
            )
            return {
                "combo_id": combo_id,
                "combo_json": combo_json,
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
                "expectancy_by_pattern": expectancy_by_pattern,
                "timeonly_wait_pct": 0.0,
                "setups_expired_pct": 0.0,
                "rr_avg": rr_avg,
                "rr_median": rr_median,
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {"combo_id": combo_id, "combo_json": combo_json, **combo, "error": str(exc)}

    total = len(combos)
    rows: list[Dict[str, Any]] = []
    best_pnl = float("-inf")
    start = time.time()

    def _fmt_eta(seconds: float) -> str:
        if seconds <= 0 or not math.isfinite(seconds):
            return "0s"
        return str(timedelta(seconds=int(seconds)))

    progress = Progress(
        TextColumn("{task.completed}/{task.total}"),
        BarColumn(),
        TextColumn("best {task.fields[best]:.2f}"),
        TextColumn("eta {task.fields[eta]}", justify="right"),
        console=Console(force_terminal=True, color_system=None),
        transient=False,
    )
    task = progress.add_task("grid", total=total, best=0.0, eta="0s")
    throttle = max(1, total // 100)

    with progress:
        if jobs <= 1:
            for i, combo in enumerate(combos):
                res = _single(i, combo)
                rows.append(res)
                processed = len(rows)
                best_pnl = max(best_pnl, res.get("pnl_net", float("-inf")))
                elapsed = time.time() - start
                eta = _fmt_eta(elapsed / processed * (total - processed)) if processed else "0s"
                if processed % throttle == 0 or processed == total:
                    progress.update(task, advance=1, best=best_pnl, eta=eta)
                else:
                    progress.advance(task, 1)
        else:
            chunksize = max(1, len(combos) // (jobs * 4))
            results_iter = Parallel(
                n_jobs=jobs,
                return_as="generator",
                batch_size=chunksize,
                **{"maxtasksperchild": 200},
            )(delayed(_single)(i, combo) for i, combo in enumerate(combos))
            for res in results_iter:
                rows.append(res)
                processed = len(rows)
                best_pnl = max(best_pnl, res.get("pnl_net", float("-inf")))
                elapsed = time.time() - start
                eta = _fmt_eta(elapsed / processed * (total - processed)) if processed else "0s"
                if processed % throttle == 0 or processed == total:
                    progress.update(task, advance=1, best=best_pnl, eta=eta)
                else:
                    progress.advance(task, 1)

    out = pd.DataFrame(rows)
    if results_path:
        to_save = out.copy()
        if "expectancy_by_pattern" in to_save.columns:
            to_save.loc[:, "expectancy_by_pattern"] = to_save["expectancy_by_pattern"].apply(
                lambda v: json.dumps(v)
            )
        atomic_to_csv(to_save, results_path)
    if meta_path:
        atomic_write_json(meta, meta_path)
    # restore indicators
    _ind.atr = _atr_orig
    _ind.ema = _ema_orig
    return out, meta
