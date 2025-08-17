from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from forest5.config import BacktestSettings
from forest5.backtest.engine import run_backtest
from forest5.backtest.grid import run_grid
from forest5.utils.io import read_ohlc_csv


# ---------------------------- CSV loading helpers ----------------------------


def load_ohlc_csv(
    path: str | Path, time_col: Optional[str] = None, sep: Optional[str] = None
) -> pd.DataFrame:
    return read_ohlc_csv(path, time_col=time_col, sep=sep)


# ------------------------------- CLI commands --------------------------------


def _parse_range(spec: str) -> Iterable[int]:
    """
    Formaty:
      - pojedyncze wartości oddzielone przecinkami: 5,8,13
      - zakres: start:stop:step (np. 5:20:1)
    """
    spec = spec.strip()
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError("Zakres musi być w formacie start:stop[:step]")
        start, stop = parts[0], parts[1]
        step = parts[2] if len(parts) == 3 else 1
        return list(range(start, stop + 1, step))
    return [int(x) for x in spec.split(",") if x]


def cmd_backtest(args: argparse.Namespace) -> int:
    df = load_ohlc_csv(args.csv, time_col=args.time_col, sep=args.sep)

    settings = BacktestSettings(
        symbol=args.symbol or "SYMBOL",
        strategy={
            "name": "ema_cross",
            "fast": args.fast,
            "slow": args.slow,
            "use_rsi": bool(args.use_rsi),
            "rsi_period": args.rsi_period,
            "rsi_oversold": args.rsi_oversold,
            "rsi_overbought": args.rsi_overbought,
        },
        risk={
            "initial_capital": float(args.capital),
            "risk_per_trade": float(args.risk),
            "max_drawdown": float(args.max_dd),
            "fee_perc": float(args.fee),
            "slippage_perc": float(args.slippage),
        },
        atr_period=args.atr_period,
        atr_multiple=args.atr_multiple,
    )

    res = run_backtest(
        df,
        settings,
        symbol=args.symbol or "SYMBOL",
        price_col="close",
        atr_period=args.atr_period,
        atr_multiple=args.atr_multiple,
    )

    # stdout: prosty podgląd najważniejszych metryk
    print(
        f"Equity end: {res.equity_curve.iloc[-1]:.2f} | "
        f"MaxDD: {res.max_dd:.3f} | Trades: {len(res.trades.trades)}"
    )

    if args.export_equity:
        out = Path(args.export_equity)
        res.equity_curve.to_csv(out, index_label="time")
        print(f"Zapisano equity do: {out.resolve()}")

    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    df = load_ohlc_csv(args.csv, time_col=args.time_col, sep=args.sep)

    fast_vals = list(_parse_range(args.fast_values))
    slow_vals = list(_parse_range(args.slow_values))

    out = run_grid(
        df,
        symbol=args.symbol or "SYMB",
        fast_values=fast_vals,
        slow_values=slow_vals,
        capital=float(args.capital),
        risk=float(args.risk),
        n_jobs=int(args.jobs),
    )

    # sortuj wg RAR / Sharpe jeśli dostępne, inaczej equity_end
    sort_cols = [c for c in ("rar", "sharpe", "equity_end") if c in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, ascending=False)

    head = out.head(args.top)
    print(head)

    if args.export:
        out_path = Path(args.export)
        if out_path.suffix.lower() in (".parquet", ".pq"):
            out.to_parquet(out_path, index=False)
        else:
            out.to_csv(out_path, index=False)
        print(f"Zapisano wyniki grid do: {out_path.resolve()}")

    return 0


# --------------------------------- Parser ------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forest5", description="FOREST 5.0 CLI")
    sub = p.add_subparsers(dest="command")

    # backtest
    p_bt = sub.add_parser("backtest", help="Uruchom pojedynczy backtest")
    p_bt.add_argument("--csv", required=True, help="Ścieżka do pliku CSV z danymi OHLC")
    p_bt.add_argument("--time-col", default=None, help="Nazwa kolumny czasu (opcjonalnie)")
    p_bt.add_argument(
        "--sep",
        default=None,
        help="Separator CSV (np. ';'). Brak = autodetekcja",
    )
    p_bt.add_argument("--symbol", default="SYMBOL", help="Symbol (np. EURUSD)")
    p_bt.add_argument("--fast", type=int, default=12, help="Szybka EMA")
    p_bt.add_argument("--slow", type=int, default=26, help="Wolna EMA")

    p_bt.add_argument("--use-rsi", action="store_true", help="Włącz filtr RSI")
    p_bt.add_argument("--rsi-period", type=int, default=14)
    p_bt.add_argument("--rsi-oversold", type=float, default=30.0)
    p_bt.add_argument("--rsi-overbought", type=float, default=70.0)

    p_bt.add_argument("--capital", type=float, default=100_000.0)
    p_bt.add_argument("--risk", type=float, default=0.01, help="Ryzyko na trade (0-1)")
    p_bt.add_argument("--max-dd", type=float, default=0.30, help="Dozwolone obsunięcie")
    p_bt.add_argument("--fee", type=float, default=0.0005, help="Prowizja %")
    p_bt.add_argument("--slippage", type=float, default=0.0, help="Poślizg %")

    p_bt.add_argument("--atr-period", type=int, default=14)
    p_bt.add_argument("--atr-multiple", type=float, default=2.0)

    p_bt.add_argument("--export-equity", default=None, help="Zapisz equity do CSV")
    p_bt.set_defaults(func=cmd_backtest)

    # grid
    p_gr = sub.add_parser("grid", help="Przeszukiwanie parametrów")
    p_gr.add_argument("--csv", required=True, help="Ścieżka do pliku CSV z danymi OHLC")
    p_gr.add_argument("--time-col", default=None, help="Nazwa kolumny czasu (opcjonalnie)")
    p_gr.add_argument(
        "--sep",
        default=None,
        help="Separator CSV (np. ';'). Brak = autodetekcja",
    )
    p_gr.add_argument("--symbol", default="SYMB", help="Symbol (np. EURUSD)")
    p_gr.add_argument("--fast-values", required=True, help="Np. 5:20:1 lub 5,8,13")
    p_gr.add_argument("--slow-values", required=True, help="Np. 10:60:2 lub 12,26")
    p_gr.add_argument("--capital", type=float, default=100_000.0)
    p_gr.add_argument("--risk", type=float, default=0.01)
    p_gr.add_argument("--jobs", type=int, default=1, help="Równoległość (1 = sekwencyjnie)")
    p_gr.add_argument("--top", type=int, default=20, help="Ile rekordów wyświetlić")
    p_gr.add_argument("--export", default=None, help="Zapis do CSV/Parquet")
    p_gr.set_defaults(func=cmd_grid)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
