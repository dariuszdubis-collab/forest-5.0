"""Walk-forward evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple
import json

import pandas as pd
from pandas.tseries.offsets import DateOffset

from ..config import BacktestSettings, StrategySettings, RiskSettings
from .engine import run_backtest as _run_backtest


def _mk_settings(
    *,
    symbol: str,
    fast: int,
    slow: int,
    use_rsi: bool,
    rsi_period: int,
    rsi_oversold: int,
    rsi_overbought: int,
    capital: float,
    risk: float,
    fee_perc: float,
    slippage_perc: float,
    atr_period: int,
    atr_multiple: float,
) -> BacktestSettings:
    strat = StrategySettings(
        name="ema_cross",
        fast=int(fast),
        slow=int(slow),
        use_rsi=bool(use_rsi),
        rsi_period=int(rsi_period),
        rsi_oversold=int(rsi_oversold),
        rsi_overbought=int(rsi_overbought),
    )
    riskset = RiskSettings(
        initial_capital=float(capital),
        risk_per_trade=float(risk),
        fee_perc=float(fee_perc),
        slippage_perc=float(slippage_perc),
    )
    return BacktestSettings(
        symbol=symbol,
        strategy=strat,
        risk=riskset,
        atr_period=int(atr_period),
        atr_multiple=float(atr_multiple),
    )


def _evaluate_df(
    df: pd.DataFrame,
    settings: BacktestSettings,
    run_backtest_func: Callable[[pd.DataFrame, BacktestSettings], object],
) -> Tuple[float, float, int, float, float, float]:
    res = run_backtest_func(df, settings)
    equity = res.equity_curve
    eq_end = float(equity.iloc[-1]) if not equity.empty else 0.0
    init_cap = float(settings.risk.initial_capital)
    ret = (eq_end / init_cap) - 1.0 if init_cap > 0 else 0.0
    pnl_net = eq_end - init_cap
    returns = equity.pct_change().dropna()
    sharpe = (
        float(returns.mean() / returns.std() * (252**0.5))
        if not returns.empty and returns.std() != 0
        else 0.0
    )
    max_dd = float(res.max_dd)
    trades = len(res.trades.trades)
    return ret, max_dd, trades, eq_end, pnl_net, sharpe


def _iter_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
    mode: str,
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    cur = start
    if mode == "anchored":
        train_start = start
        train_end = train_start + DateOffset(months=train_months) - pd.Timedelta(seconds=1)
        while True:
            test_start = train_end + pd.Timedelta(seconds=1)
            test_end = test_start + DateOffset(months=test_months) - pd.Timedelta(seconds=1)
            if train_end > end or test_start > end:
                break
            yield (train_start, train_end, test_start, min(test_end, end))
            train_end = train_end + DateOffset(months=step_months)
            cur = cur + DateOffset(months=step_months)
            if cur > end:
                break
    else:  # rolling
        while True:
            train_start = cur
            train_end = train_start + DateOffset(months=train_months) - pd.Timedelta(seconds=1)
            test_start = train_end + pd.Timedelta(seconds=1)
            test_end = test_start + DateOffset(months=test_months) - pd.Timedelta(seconds=1)
            if train_end > end or test_start > end:
                break
            yield (train_start, train_end, test_start, min(test_end, end))
            cur = cur + DateOffset(months=step_months)
            if cur > end:
                break


def run_walkforward(
    df: pd.DataFrame,
    *,
    symbol: str,
    fast_values: Iterable[int],
    slow_values: Iterable[int],
    use_rsi: bool = False,
    rsi_period: int = 14,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
    capital: float = 100_000.0,
    risk: float = 0.01,
    fee_perc: float = 0.0005,
    slippage_perc: float = 0.0,
    atr_period: int = 14,
    atr_multiple: float = 2.0,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
    mode: str = "rolling",
    skip_fast_ge_slow: bool = False,
    out_dir: str | Path | None = None,
    run_backtest_func: Callable[[pd.DataFrame, BacktestSettings], object] = _run_backtest,
) -> pd.DataFrame:
    """Run walk-forward analysis on ``df``.

    Parameters are intentionally similar to :mod:`scripts.walkforward`.
    """

    start_ts = df.index[0]
    end_ts = df.index[-1]

    grid: List[Dict[str, int | bool]] = []
    for f in fast_values:
        for s in slow_values:
            if skip_fast_ge_slow and f >= s:
                continue
            grid.append(
                {
                    "fast": int(f),
                    "slow": int(s),
                    "use_rsi": bool(use_rsi),
                    "rsi_period": int(rsi_period),
                    "rsi_oversold": int(rsi_oversold),
                    "rsi_overbought": int(rsi_overbought),
                }
            )

    out_path: Path | None = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for i, (tr_start, tr_end, te_start, te_end) in enumerate(
        _iter_windows(start_ts, end_ts, train_months, test_months, step_months, mode)
    ):
        train_df = df.loc[(df.index >= tr_start) & (df.index <= tr_end)]
        test_df = df.loc[(df.index >= te_start) & (df.index <= te_end)]
        if train_df.empty or test_df.empty:
            continue

        best: Dict[str, int | bool] | None = None
        best_pnl = -1e18
        best_sharpe = -1e18
        for p in grid:
            st = _mk_settings(
                symbol=symbol,
                fast=int(p["fast"]),
                slow=int(p["slow"]),
                use_rsi=bool(p["use_rsi"]),
                rsi_period=int(p["rsi_period"]),
                rsi_oversold=int(p["rsi_oversold"]),
                rsi_overbought=int(p["rsi_overbought"]),
                capital=capital,
                risk=risk,
                fee_perc=fee_perc,
                slippage_perc=slippage_perc,
                atr_period=atr_period,
                atr_multiple=atr_multiple,
            )
            _, _, _, _, pnl_net, sharpe = _evaluate_df(train_df, st, run_backtest_func)
            if (pnl_net > best_pnl) or (pnl_net == best_pnl and sharpe > best_sharpe):
                best = p
                best_pnl = pnl_net
                best_sharpe = sharpe
        if best is None:
            continue

        best_settings = _mk_settings(
            symbol=symbol,
            fast=int(best["fast"]),
            slow=int(best["slow"]),
            use_rsi=bool(best["use_rsi"]),
            rsi_period=int(best["rsi_period"]),
            rsi_oversold=int(best["rsi_oversold"]),
            rsi_overbought=int(best["rsi_overbought"]),
            capital=capital,
            risk=risk,
            fee_perc=fee_perc,
            slippage_perc=slippage_perc,
            atr_period=atr_period,
            atr_multiple=atr_multiple,
        )
        te_ret, te_dd, te_trades, te_eq_end, te_pnl_net, te_sharpe = _evaluate_df(
            test_df, best_settings, run_backtest_func
        )

        wf_score = te_pnl_net / best_pnl if best_pnl else 0.0

        row = {
            "fold": i,
            "train_start": tr_start,
            "train_end": tr_end,
            "test_start": te_start,
            "test_end": te_end,
            "fast": int(best["fast"]),
            "slow": int(best["slow"]),
            "use_rsi": bool(best["use_rsi"]),
            "rsi_period": int(best["rsi_period"]),
            "rsi_oversold": int(best["rsi_oversold"]),
            "rsi_overbought": int(best["rsi_overbought"]),
            "train_pnl_net": best_pnl,
            "train_sharpe": best_sharpe,
            "test_ret": te_ret,
            "test_max_dd": te_dd,
            "test_trades": te_trades,
            "test_equity_end": te_eq_end,
            "test_pnl_net": te_pnl_net,
            "test_sharpe": te_sharpe,
            "wf_score": wf_score,
        }
        rows.append(row)

        if out_path:
            with (out_path / f"fold_{i}.json").open("w", encoding="utf-8") as fh:
                json.dump(row, fh, default=str, ensure_ascii=False, indent=2)

    result_df = pd.DataFrame(rows)
    if out_path:
        result_df.to_csv(out_path / "summary.csv", index=False)
    return result_df
