#!/usr/bin/env python
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forest5.config import BacktestSettings, StrategySettings, RiskSettings
from forest5.backtest.engine import run_backtest
from forest5.utils.validate import ensure_backtest_ready

def parse_args():
    p = argparse.ArgumentParser(description="MTM guard / sanity tracer")
    p.add_argument("--csv", required=True)
    p.add_argument("--symbol", default="SYMBOL")
    p.add_argument("--fast", type=int, default=12)
    p.add_argument("--slow", type=int, default=26)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--atr-multiple", type=float, default=2.0)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--out", default="mtm_guard.csv")
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = ensure_backtest_ready(df, price_col="close")

    if args.start:
        df = df[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end)]

    s = BacktestSettings(
        symbol=args.symbol,
        strategy=StrategySettings(name="ema_cross", fast=args.fast, slow=args.slow, use_rsi=False),
        risk=RiskSettings(initial_capital=args.capital, risk_per_trade=args.risk, fee_perc=0.0, slippage_perc=0.0),
        atr_period=args.atr_period, atr_multiple=args.atr_multiple
    )
    res = run_backtest(df, s)

    bars = len(df)
    marks = len(res.equity_curve)
    mean_eq = float(np.mean(res.equity_curve)) if marks else float("nan")

    # Heurystyki
    ok_len = marks in (bars, bars + 1)
    ok_scale = mean_eq > args.capital * 0.1  # equity nie może być w skali ceny

    # Dopasowanie timestampów do equity (N lub N+1)
    if marks == bars + 1:
        ts = pd.Index([df.index[0]]).append(df.index)  # start + każdy bar
    elif marks == bars:
        ts = df.index
    else:
        # fallback: liczbowy index
        ts = pd.RangeIndex(marks)

    out = Path(args.out)
    out_df = pd.DataFrame({"time": ts, "equity": res.equity_curve.values})
    out_df.to_csv(out, index=False)

    summary = {
        "bars": bars,
        "equity_marks": marks,
        "ok_len": ok_len,
        "ok_scale": ok_scale,
        "equity_mean": mean_eq,
        "max_dd": res.max_dd,
        "out": str(out),
    }
    print(summary)

    # Jeżeli coś nie gra – sygnalizujemy błędem procesu (łatwo łapać w CI/Makefile)
    if not (ok_len and ok_scale):
        sys.exit(1)

if __name__ == "__main__":
    main()

