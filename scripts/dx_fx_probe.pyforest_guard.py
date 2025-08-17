from __future__ import annotations
import argparse
import pandas as pd

from forest5.config import BacktestSettings, StrategySettings, RiskSettings
from forest5.utils.validate import ensure_backtest_ready
from forest5.backtest.engine import run_backtest
from forest5.utils.argparse_ext import PercentAction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--fast", type=int, default=12)
    p.add_argument("--slow", type=int, default=26)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--atr-multiple", type=float, default=2.0)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--risk", action=PercentAction, default=0.01)
    p.add_argument("--fee-perc", action=PercentAction, default=0.0005)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--inspect-n", type=int, default=0)
    p.add_argument("--out", default="dx_debug.csv")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    # normalizacja czasu
    for c in ("time", "date", "datetime", "timestamp"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=False, errors="coerce")
            df = df.set_index(c)
            break

    df = ensure_backtest_ready(df)

    if args.start:
        df = df.loc[args.start :]
    if args.end:
        df = df.loc[: args.end]

    if args.inspect_n:
        head = df.head(args.inspect_n).copy()
        head.to_csv(args.out, index=True)
        print({"rows": len(head), "out": args.out, "event": "dx_export_done"})

    settings = BacktestSettings(
        symbol=args.symbol,
        strategy=StrategySettings(name="ema_cross", fast=args.fast, slow=args.slow, use_rsi=False),
        risk=RiskSettings(
            initial_capital=args.capital,
            risk_per_trade=args.risk,
            fee_perc=args.fee_perc,
            slippage_perc=0.0,
        ),
        atr_period=args.atr_period,
        atr_multiple=args.atr_multiple,
    )

    res = run_backtest(df, settings)
    print(
        {
            "equity_end": float(res.equity_curve.iloc[-1]),
            "max_dd": float(res.max_dd),
            "trades": len(res.trades.trades),
            "event": "backtest_done",
        }
    )


if __name__ == "__main__":
    main()
