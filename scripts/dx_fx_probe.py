#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import pandas as pd

from forest5.config import BacktestSettings, StrategySettings, RiskSettings
from forest5.utils.validate import ensure_backtest_ready
from forest5.backtest.engine import run_backtest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("dx_fx_probe")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--fast", type=int, default=12)
    ap.add_argument("--slow", type=int, default=26)
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--atr-multiple", type=float, default=2.0)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--fee-perc", type=float, default=0.0005)
    ap.add_argument("--slippage-perc", type=float, default=0.0)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--inspect-n", type=int, default=0)
    ap.add_argument("--out", type=str, default="dx_debug.csv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = ensure_backtest_ready(df, price_col="close")

    if args.start:
        df = df[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end)]

    settings = BacktestSettings(
        symbol=args.symbol,
        strategy=StrategySettings(name="ema_cross", fast=args.fast, slow=args.slow, use_rsi=False),
        risk=RiskSettings(
            initial_capital=args.capital,
            risk_per_trade=args.risk,
            max_drawdown=0.30,
            fee_perc=args.fee_perc,
            slippage_perc=args.slippage_perc,
        ),
        atr_period=args.atr_period,
        atr_multiple=args.atr_multiple,
    )

    res = run_backtest(df, settings)

    # diagnostyka – eksport krzywej equity M2M
    if args.inspect_n and args.inspect_n > 0:
        n = min(args.inspect_n, len(res.equity_curve))
        out_df = pd.DataFrame({"equity_mtm": res.equity_curve.iloc[:n].values})
        out_df.to_csv(args.out, index=False)
        print(json.dumps({"rows": len(out_df), "out": args.out, "event": "dx_export_done"}))

    print(json.dumps({
        "equity_end": float(res.equity_curve.iloc[-1]),
        "max_dd": float(res.max_dd),
        "trades": len(res.trades.trades),
        "event": "backtest_done"
    }))


if __name__ == "__main__":
    main()

