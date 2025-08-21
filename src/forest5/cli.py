from __future__ import annotations

import re
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from forest5.config import BacktestSettings
from forest5.config_live import LiveSettings
from forest5.backtest.engine import run_backtest
from forest5.backtest.grid import run_grid
from forest5.live.live_runner import run_live
from forest5.utils.io import read_ohlc_csv
from forest5.utils.argparse_ext import PercentAction


class SafeHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Help formatter that escapes bare '%' to avoid ValueError in argparse."""

    def _expand_help(self, action):
        params = dict(vars(action), prog=self._prog)
        help_text = self._get_help_string(action) or ""
        # Replace bare '%' with '%%' so argparse doesn't treat it as a
        # formatting placeholder. We intentionally ignore patterns like
        # '%(foo)s', which argparse uses to inject defaults.
        help_text = re.sub(r"%(?!\()", "%%", help_text)
        return help_text % params


# ---------------------------- CSV loading helpers ----------------------------


def load_ohlc_csv(
    path: str | Path, time_col: Optional[str] = None, sep: Optional[str] = None
) -> pd.DataFrame:
    return read_ohlc_csv(path, time_col=time_col, sep=sep)


# ------------------------------- CLI commands --------------------------------


def _parse_span_or_list(spec: str) -> list[int]:
    """Parse a numeric span (``lo-hi[:step]``) or comma-separated list.

    Examples::
        "5-7"        -> [5, 6, 7]
        "1-5:2"      -> [1, 3, 5]
        "1,2,10"     -> [1, 2, 10]

    The range is inclusive and supports negative numbers, e.g. ``-3--1``.
    Raises :class:`argparse.ArgumentTypeError` when the specification is
    malformed, ``lo > hi`` or ``step <= 0``.
    """

    spec = str(spec).strip()

    # Comma separated list of values
    if "," in spec:
        try:
            return [int(float(x.strip())) for x in spec.split(",") if x.strip()]
        except ValueError as ex:
            raise argparse.ArgumentTypeError(f"Invalid list: {spec}") from ex

    # Extract optional step (lo-hi[:step])
    core, step_str = (spec.split(":", 1) + ["1"])[:2]
    try:
        step = int(float(step_str))
    except ValueError as ex:
        raise argparse.ArgumentTypeError(f"Invalid step: {step_str}") from ex
    if step <= 0:
        raise argparse.ArgumentTypeError(f"Step must be > 0 (given: {step})")

    # Single value without span
    import re

    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*", core)
    if not m:
        try:
            return [int(float(core))]
        except ValueError as ex:
            raise argparse.ArgumentTypeError(f"Invalid range: {spec}") from ex

    lo = int(float(m.group(1)))
    hi = int(float(m.group(2)))
    if hi < lo:
        raise argparse.ArgumentTypeError(f"Upper bound < lower: {spec}")

    vals: list[int] = list(range(lo, hi + 1, step))
    if vals[-1] != hi:
        vals.append(hi)
    return vals


# Backwards compatibility – old name used in previous versions/tests
_parse_range = _parse_span_or_list


def _parse_int_list(spec: str | None) -> list[int]:
    """Parse comma-separated integers into a list of ints."""
    if not spec:
        return []
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


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

    settings.time.use_time_model = bool(args.time_model)
    settings.time.time_model_path = args.time_model
    settings.time.fusion_min_confluence = int(args.min_confluence)
    settings.time.blocked_hours = args.blocked_hours or []
    settings.time.blocked_weekdays = args.blocked_weekdays or []

    res = run_backtest(
        df,
        settings,
        symbol=args.symbol or "SYMBOL",
        price_col="close",
        atr_period=args.atr_period,
        atr_multiple=args.atr_multiple,
    )

    equity = res.equity_curve
    equity_end = float(equity.iloc[-1]) if not equity.empty else 0.0
    ret = equity_end / float(settings.risk.initial_capital) - 1.0

    # stdout: prosty podgląd najważniejszych metryk
    print(
        f"Equity end: {equity_end:.2f} | "
        f"Return: {ret:.6f} | "
        f"MaxDD: {res.max_dd:.3f} | Trades: {len(res.trades.trades)}"
    )

    if args.export_equity:
        out = Path(args.export_equity)
        res.equity_curve.to_csv(out, index_label="time")
        print(f"Zapisano equity do: {out.resolve()}")

    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    df = load_ohlc_csv(args.csv, time_col=args.time_col, sep=args.sep)

    fast_vals = list(_parse_span_or_list(args.fast_values))
    slow_vals = list(_parse_span_or_list(args.slow_values))

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


def cmd_live(args: argparse.Namespace) -> int:
    settings = LiveSettings.from_file(args.config)
    if args.paper:
        settings.broker.type = "paper"
    run_live(settings)
    return 0


# --------------------------------- Parser ------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forest5",
        description="Forest 5.0 – modularny framework tradingowy.",
        formatter_class=SafeHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # backtest
    p_bt = sub.add_parser(
        "backtest", help="Uruchom pojedynczy backtest", formatter_class=SafeHelpFormatter
    )
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
    p_bt.add_argument("--rsi-oversold", type=int, default=30, choices=range(0, 101))
    p_bt.add_argument("--rsi-overbought", type=int, default=70, choices=range(0, 101))

    p_bt.add_argument("--capital", type=float, default=100_000.0)
    p_bt.add_argument("--risk", action=PercentAction, default=0.01, help="Ryzyko na trade (0-1)")
    p_bt.add_argument("--max-dd", action=PercentAction, default=0.30, help="Dozwolone obsunięcie")
    p_bt.add_argument("--fee", action=PercentAction, default=0.0005, help="Prowizja %")
    p_bt.add_argument("--slippage", action=PercentAction, default=0.0, help="Poślizg %")

    p_bt.add_argument("--atr-period", type=int, default=14)
    p_bt.add_argument("--atr-multiple", type=float, default=2.0)

    p_bt.add_argument("--time-model", type=Path, default=None, help="Ścieżka do modelu czasu")
    p_bt.add_argument("--min-confluence", type=int, default=1, help="Minimalna konfluencja fuzji")
    p_bt.add_argument(
        "--blocked-hours",
        type=_parse_int_list,
        default=None,
        help="Zablokowane godziny (0-23), np. 0,1,2",
    )
    p_bt.add_argument(
        "--blocked-weekdays",
        type=_parse_int_list,
        default=None,
        help="Zablokowane dni tygodnia 0=Mon..6=Sun",
    )

    p_bt.add_argument("--export-equity", default=None, help="Zapisz equity do CSV")
    p_bt.set_defaults(func=cmd_backtest)

    # grid
    p_gr = sub.add_parser(
        "grid", help="Przeszukiwanie parametrów", formatter_class=SafeHelpFormatter
    )
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
    p_gr.add_argument("--risk", action=PercentAction, default=0.01)
    p_gr.add_argument("--jobs", type=int, default=1, help="Równoległość (1 = sekwencyjnie)")
    p_gr.add_argument("--top", type=int, default=20, help="Ile rekordów wyświetlić")
    p_gr.add_argument("--export", default=None, help="Zapis do CSV/Parquet")
    p_gr.set_defaults(func=cmd_grid)

    # live
    p_lv = sub.add_parser("live", help="Uruchom trading na żywo", formatter_class=SafeHelpFormatter)
    p_lv.add_argument("--config", required=True, help="Ścieżka do pliku YAML/JSON z ustawieniami")
    p_lv.add_argument("--paper", action="store_true", help="Wymuś paper trading (broker.type)")
    p_lv.set_defaults(func=cmd_live)

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
