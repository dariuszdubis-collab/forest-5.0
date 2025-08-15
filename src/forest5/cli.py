from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .backtest.engine import run_backtest
from .backtest.grid import run_grid
from .config import BacktestSettings
from .examples.synthetic import generate_ohlc


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=False)
        df = df.set_index("time")
    return df


def cmd_backtest(args: argparse.Namespace) -> None:
    df = _load_csv(Path(args.data))
    settings = BacktestSettings(
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy=dict(name="ema_cross", fast=args.fast, slow=args.slow),
        risk=dict(
            initial_capital=args.capital,
            risk_per_trade=args.risk,
            max_drawdown=args.max_dd,
            fee_perc=args.fee,
            slippage_perc=args.slip,
        ),
        atr_period=args.atr,
        atr_multiple=args.atr_mult,
    )
    res = run_backtest(df, settings)
    eq = res.equity_curve
    print("Backtest finished:")
    print(f"- points: {len(eq)}")
    print(f"- equity_end: {float(eq.iloc[-1]):.2f}")
    print(f"- max_dd: {res.max_dd:.2%}")


def cmd_grid(args: argparse.Namespace) -> None:
    df = _load_csv(Path(args.data))
    fast_values = list(range(args.fast_start, args.fast_stop + 1, args.fast_step))
    slow_values = list(range(args.slow_start, args.slow_stop + 1, args.slow_step))
    df_res = run_grid(
        df,
        symbol=args.symbol,
        fast_values=fast_values,
        slow_values=slow_values,
        capital=args.capital,
        risk=args.risk,
        max_dd=args.max_dd,
        atr_period=args.atr,
        atr_multiple=args.atr_mult,
        n_jobs=args.jobs,
        cache_dir=args.cache,
    )
    print(df_res.sort_values("equity_end", ascending=False).head(10))


def cmd_demo(args: argparse.Namespace) -> None:
    df = generate_ohlc(periods=args.periods, start_price=args.start, freq=args.freq)
    if args.out:
        df.to_csv(args.out, index=True, date_format="%Y-%m-%d %H:%M:%S")
        print(f"Saved demo CSV -> {args.out}")
    else:
        print(df.head(10))


def main() -> None:
    p = argparse.ArgumentParser("forest5")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="Run single backtest on CSV")
    b.add_argument("--data", required=True)
    b.add_argument("--symbol", default="SYMBOL")
    b.add_argument("--timeframe", default="1h")
    b.add_argument("--fast", type=int, default=12)
    b.add_argument("--slow", type=int, default=26)
    b.add_argument("--capital", type=float, default=100_000.0)
    b.add_argument("--risk", type=float, default=0.01)
    b.add_argument("--max-dd", type=float, default=0.30)
    b.add_argument("--fee", type=float, default=0.0005)
    b.add_argument("--slip", type=float, default=0.0)
    b.add_argument("--atr", type=int, default=14)
    b.add_argument("--atr-mult", type=float, default=2.0)
    b.set_defaults(func=cmd_backtest)

    g = sub.add_parser("grid", help="Grid-search for EMA fast/slow")
    g.add_argument("--data", required=True)
    g.add_argument("--symbol", default="SYMBOL")
    g.add_argument("--capital", type=float, default=100_000.0)
    g.add_argument("--risk", type=float, default=0.01)
    g.add_argument("--max-dd", type=float, default=0.30)
    g.add_argument("--atr", type=int, default=14)
    g.add_argument("--atr-mult", type=float, default=2.0)
    g.add_argument("--fast-start", type=int, default=6)
    g.add_argument("--fast-stop", type=int, default=18)
    g.add_argument("--fast-step", type=int, default=6)
    g.add_argument("--slow-start", type=int, default=20)
    g.add_argument("--slow-stop", type=int, default=60)
    g.add_argument("--slow-step", type=int, default=10)
    g.add_argument("--jobs", type=int, default=1)
    g.add_argument("--cache", default=".cache/forest5-grid")
    g.set_defaults(func=cmd_grid)

    d = sub.add_parser("demo", help="Generate synthetic OHLC data")
    d.add_argument("--periods", type=int, default=50)
    d.add_argument("--start", type=float, default=100.0)
    d.add_argument("--freq", default="D")
    d.add_argument("--out")
    d.set_defaults(func=cmd_demo)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

