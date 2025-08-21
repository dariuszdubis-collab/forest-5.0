#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from forest5.config import BacktestSettings, RiskSettings, StrategySettings
from forest5.backtest.engine import run_backtest
from forest5.utils.validate import ensure_backtest_ready
from forest5.utils.argparse_ext import PercentAction


# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------


def log_json(**payload: object) -> None:
    """Emit one JSON line (UTF-8, no ASCII escaping)."""
    print(json.dumps(payload, ensure_ascii=False))


def parse_range(spec: str) -> list[int]:
    """
    Parse a numeric range like:
      "5-20"        -> [5, 6, ..., 20]
      "20-60:5"     -> [20, 25, 30, ..., 60]
      "10"          -> [10]
    """
    spec = str(spec).strip()
    if "-" not in spec:
        # single value
        try:
            return [int(float(spec))]
        except ValueError as ex:
            raise argparse.ArgumentTypeError(f"Niepoprawny zakres: {spec}") from ex

    core, step_str = (spec.split(":", 1) + ["1"])[:2]
    lo_str, hi_str = core.split("-", 1)
    lo = int(float(lo_str))
    hi = int(float(hi_str))
    step = int(float(step_str))
    if step <= 0:
        raise argparse.ArgumentTypeError(f"Krok musi być > 0 (given: {step})")
    if hi < lo:
        raise argparse.ArgumentTypeError(f"Górna granica < dolnej: {spec}")

    # inclusive range
    vals: list[int] = list(range(lo, hi + 1, step))
    if vals[-1] != hi:
        vals.append(hi)
    return vals


def _parse_int_list(spec: str | None) -> list[int]:
    """Parse comma-separated integers into a list."""
    if not spec:
        return []
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


@dataclass(frozen=True)
class GridPoint:
    fast: int
    slow: int
    use_rsi: bool
    rsi_period: int
    rsi_oversold: int
    rsi_overbought: int


@dataclass
class GridResult:
    fast: int
    slow: int
    use_rsi: bool
    rsi_period: int
    rsi_oversold: int
    rsi_overbought: int
    ret: float
    max_dd: float
    trades: int
    equity_end: float
    score: float


def _run_one(
    df: pd.DataFrame, gp: GridPoint, base: BacktestSettings, dd_penalty: float
) -> GridResult:
    """Run backtest for a single grid point."""
    # Build settings with chosen strategy parameters
    stg = StrategySettings(
        name="ema_cross",
        fast=gp.fast,
        slow=gp.slow,
        use_rsi=gp.use_rsi,
        rsi_period=gp.rsi_period,
        rsi_overbought=gp.rsi_overbought,
        rsi_oversold=gp.rsi_oversold,
    )
    s = BacktestSettings(
        symbol=base.symbol,
        timeframe=base.timeframe,
        strategy=stg,
        risk=base.risk,
        time=base.time,
        atr_period=base.atr_period,
        atr_multiple=base.atr_multiple,
    )
    res = run_backtest(df, s)

    equity = res.equity_curve
    equity_end = float(equity.iloc[-1]) if not equity.empty else 0.0
    initial = float(s.risk.initial_capital)
    ret = equity_end / initial - 1.0
    max_dd = float(res.max_dd)
    trades = len(res.trades.trades)
    score = ret - dd_penalty * max_dd
    return GridResult(
        fast=gp.fast,
        slow=gp.slow,
        use_rsi=gp.use_rsi,
        rsi_period=gp.rsi_period,
        rsi_oversold=gp.rsi_oversold,
        rsi_overbought=gp.rsi_overbought,
        ret=ret,
        max_dd=max_dd,
        trades=trades,
        equity_end=equity_end,
        score=score,
    )


def _load_csv(csv_path: str | os.PathLike[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Let validate helper detect/normalize OHLC/time
    df.index.name = None
    df = ensure_backtest_ready(df)
    return df


def _filter_time(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


def _export_csv(path: Path, rows: list[GridResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "fast",
                "slow",
                "use_rsi",
                "rsi_period",
                "rsi_oversold",
                "rsi_overbought",
                "ret",
                "max_dd",
                "trades",
                "equity_end",
                "score",
            ],
        )
        w.writeheader()
        for r in rows:
            d = asdict(r)
            # Normalize floats for CSV readability
            d["ret"] = f"{d['ret']:.6f}"
            d["max_dd"] = f"{d['max_dd']:.6f}"
            d["equity_end"] = f"{d['equity_end']:.6f}"
            w.writerow(d)


def _print_top(rows: list[GridResult], n: int = 10) -> None:
    if not rows:
        return
    df = pd.DataFrame([asdict(r) for r in rows])
    df = df.sort_values("score", ascending=False).head(n)
    # Pretty print numeric columns with limited precision
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.to_string(index=False, justify="right"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forest 5.0 – optymalizacja parametrów (grid search)."
    )
    parser.add_argument("--csv", required=True, help="Ścieżka do CSV z danymi OHLC.")
    parser.add_argument("--symbol", default="SYMBOL", help="Symbol (opisowy).")
    parser.add_argument("--fast", required=True, help='Zakres np. "5-20" albo "5-20:5".')
    parser.add_argument("--slow", required=True, help='Zakres np. "20-60" albo "20-60:5".')
    parser.add_argument(
        "--skip-fast-ge-slow", action="store_true", help="Pomiń pary gdzie fast >= slow."
    )
    parser.add_argument("--use-rsi", action="store_true", help="Włącz filtr RSI.")
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rsi-oversold", type=int, default=30, choices=range(0, 101))
    parser.add_argument("--rsi-overbought", type=int, default=70, choices=range(0, 101))

    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument(
        "--risk", action=PercentAction, default=0.01, help="Udział kapitału ryzykowany na trade."
    )
    parser.add_argument("--fee-perc", action=PercentAction, default=0.0005)
    parser.add_argument("--slippage-perc", action=PercentAction, default=0.0)

    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--atr-multiple", type=float, default=2.0)

    parser.add_argument("--time-model", type=Path, default=None, help="Ścieżka do modelu czasu.")
    parser.add_argument("--min-confluence", type=int, default=1, help="Minimalna konfluencja fuzji")
    parser.add_argument(
        "--blocked-hours",
        type=_parse_int_list,
        default=None,
        help="Zablokowane godziny (0-23), np. 0,1,2",
    )
    parser.add_argument(
        "--blocked-weekdays",
        type=_parse_int_list,
        default=None,
        help="Zablokowane dni tygodnia 0=Mon..6=Sun",
    )

    parser.add_argument("--start", type=str, default=None, help="Początek zakresu (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Koniec zakresu (YYYY-MM-DD).")

    parser.add_argument("--dd-penalty", type=float, default=0.5, help="Kara za DD w funkcji celu.")
    parser.add_argument("--jobs", type=int, default=1, help="Równoległość (ProcessPool).")

    parser.add_argument("--quiet", action="store_true", help="Mniej logów.")
    parser.add_argument("--export", type=str, default=None, help="Zapisz wyniki do CSV.")

    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    # Load & trim data
    df = _load_csv(args.csv)
    df = _filter_time(df, args.start, args.end)

    # Base settings (risk etc.)
    base = BacktestSettings(
        symbol=args.symbol,
        strategy=StrategySettings(name="ema_cross", fast=12, slow=26, use_rsi=args.use_rsi),
        risk=RiskSettings(
            initial_capital=float(args.capital),
            risk_per_trade=float(args.risk),
            max_drawdown=0.3,
            fee_perc=float(args.fee_perc),
            slippage_perc=float(args.slippage_perc),
        ),
        atr_period=int(args.atr_period),
        atr_multiple=float(args.atr_multiple),
    )

    base.time.model.enabled = bool(args.time_model)
    base.time.model.path = args.time_model
    base.time.fusion_min_confluence = int(args.min_confluence)
    base.time.blocked_hours = args.blocked_hours or []
    base.time.blocked_weekdays = args.blocked_weekdays or []

    fast_vals = parse_range(args.fast)
    slow_vals = parse_range(args.slow)

    # Build grid
    points: list[GridPoint] = []
    for f in fast_vals:
        for s in slow_vals:
            if args.skip_fast_ge_slow and f >= s:
                continue
            points.append(
                GridPoint(
                    fast=int(f),
                    slow=int(s),
                    use_rsi=bool(args.use_rsi),
                    rsi_period=int(args.rsi_period),
                    rsi_oversold=int(args.rsi_oversold),
                    rsi_overbought=int(args.rsi_overbought),
                )
            )

    # Execute
    results: list[GridResult] = []
    if args.jobs and args.jobs > 1:
        # Multiprocessing – slice df once for child processes via closure pickling
        with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            futures = {
                ex.submit(_run_one, df, gp, base, float(args.dd_penalty)): gp for gp in points
            }
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for gp in points:
            results.append(_run_one(df, gp, base, float(args.dd_penalty)))

    # Export / Print
    if args.export:
        out = Path(args.export)
        _export_csv(out, results)
        log_json(event="grid_export_done", out=str(out), rows=len(results))

    # Print top table at the end
    log_json(event="grid_top", rows=min(10, len(results)))
    _print_top(results, n=10)


if __name__ == "__main__":
    main()
