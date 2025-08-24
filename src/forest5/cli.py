from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from forest5.config import (
    ALLOWED_SYMBOLS,
    BacktestSettings,
    DEFAULT_DATA_DIR,
    get_data_dir,
    load_live_settings,
)
from forest5.backtest.engine import run_backtest
from forest5.backtest.grid import run_grid
from forest5.live.live_runner import run_live
from forest5.utils.io import read_ohlc_csv, load_symbol_csv
from forest5.utils.argparse_ext import PercentAction, parse_span_or_list
from forest5.utils.log import setup_logger

_parse_span_or_list = parse_span_or_list


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


def assert_h1_ohlc(df: pd.DataFrame) -> None:
    """Ensure ``df`` contains hourly OHLC data.

    Parameters
    ----------
    df:
        Data indexed by :class:`~pandas.DatetimeIndex`.

    Raises
    ------
    ValueError
        If the index step is not 1 hour or columns differ from the
        expected ``open/high/low/close`` set with optional ``volume``.
    """

    expected = ["open", "high", "low", "close"]
    allowed = expected + ["volume"]

    cols = list(df.columns)
    if cols not in (expected, allowed):
        raise ValueError(
            "Data must contain columns 'open', 'high', 'low', 'close'" " with optional 'volume'"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex with 1H step")

    if len(df.index) > 1:
        deltas = df.index.to_series().diff().dropna()
        if not (deltas == pd.Timedelta(hours=1)).all():
            raise ValueError("Index must have 1H frequency")


def load_ohlc_csv(
    path: str | Path, time_col: Optional[str] = None, sep: Optional[str] = None
) -> pd.DataFrame:
    df = read_ohlc_csv(path, time_col=time_col, sep=sep)
    assert_h1_ohlc(df)
    return df


# Backwards compatibility – old name used in previous versions/tests
_parse_range = _parse_span_or_list


def _parse_int_list(spec: str | None) -> list[int]:
    """Parse comma-separated integers into a list of ints."""
    if not spec:
        return []
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _parse_str_list(spec: str | None) -> list[str]:
    if not spec:
        return []
    return [s.strip() for s in str(spec).split(",") if s.strip()]


def _parse_float_list(spec: str | None) -> list[float]:
    """Parse comma-separated floats into a list of floats."""
    if not spec:
        return []
    return [float(x.strip()) for x in str(spec).split(",") if x.strip()]


def cmd_backtest(args: argparse.Namespace) -> int:
    if args.csv is not None:
        csv_path = Path(args.csv)
    else:
        data_dir = get_data_dir(args.data_dir)
        csv_path = data_dir / f"{args.symbol}_H1.csv"

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    df = load_ohlc_csv(csv_path, time_col=args.time_col, sep=args.sep)

    settings = BacktestSettings(
        symbol=args.symbol,
        timeframe="1h",
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
        debug_dir=args.debug_dir,
    )

    settings.time.model.enabled = bool(args.time_model)
    settings.time.model.path = args.time_model
    settings.time.fusion_min_confluence = float(args.min_confluence)

    res = run_backtest(
        df,
        settings,
        symbol=args.symbol,
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
    kwargs: dict[str, object] = {}
    if args.strategy_name:
        kwargs["strategy_name"] = args.strategy_name
    if args.timeframe:
        kwargs["timeframe"] = args.timeframe
    if args.ema_fast:
        kwargs["ema_fast"] = [int(x) for x in args.ema_fast]
    if args.ema_slow:
        kwargs["ema_slow"] = [int(x) for x in args.ema_slow]
    if args.rsi_len:
        kwargs["rsi_len"] = [int(x) for x in args.rsi_len]
    if args.atr_len:
        kwargs["atr_len"] = [int(x) for x in args.atr_len]
    if args.t_sep_atr:
        kwargs["t_sep_atr"] = args.t_sep_atr
    if args.pullback_atr:
        kwargs["pullback_atr"] = args.pullback_atr
    if args.entry_buffer_atr:
        kwargs["entry_buffer_atr"] = args.entry_buffer_atr
    if args.sl_buffer_atr:
        kwargs["sl_buffer_atr"] = args.sl_buffer_atr
    if args.sl_min_atr:
        kwargs["sl_min_atr"] = args.sl_min_atr
    if args.tp_rr:
        kwargs["tp_rr"] = args.tp_rr
    if args.trailing_atr:
        kwargs["trailing_atr"] = args.trailing_atr
    if args.setup_ttl_bars:
        kwargs["setup_ttl_bars"] = [int(x) for x in args.setup_ttl_bars]
    if args.engulf_eps_atr:
        kwargs["engulf_eps_atr"] = args.engulf_eps_atr
    if args.engulf_body_min:
        kwargs["engulf_body_min"] = args.engulf_body_min
    if args.pinbar_wick_dom:
        kwargs["pinbar_wick_dom"] = args.pinbar_wick_dom
    if args.pinbar_body_max:
        kwargs["pinbar_body_max"] = args.pinbar_body_max
    if args.pinbar_opp_wick_max:
        kwargs["pinbar_opp_wick_max"] = args.pinbar_opp_wick_max
    if args.star_reclaim_min:
        kwargs["star_reclaim_min"] = args.star_reclaim_min
    if args.star_mid_small_max:
        kwargs["star_mid_small_max"] = args.star_mid_small_max
    if args.q_low:
        kwargs["q_low"] = [int(x) for x in args.q_low]
    if args.q_high:
        kwargs["q_high"] = [int(x) for x in args.q_high]
    kwargs["disable_timeonly"] = bool(args.disable_timeonly)
    if args.pairs:
        kwargs["pairs"] = args.pairs
    if args.from_:
        kwargs["from_"] = args.from_
    if args.to:
        kwargs["to"] = args.to
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.out:
        kwargs["out"] = Path(args.out)
    kwargs["parallel"] = bool(args.parallel)
    if args.max_workers is not None:
        kwargs["max_workers"] = args.max_workers

    run_grid(**kwargs)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    settings = load_live_settings(args.config)
    if args.paper:
        settings.broker.type = "paper"
    kwargs = {}
    if args.debug_dir:
        kwargs["debug_dir"] = args.debug_dir
    run_live(settings, **kwargs)
    return 0


# --------------------------------- Parser ------------------------------------


def add_data_source_args(parser: argparse.ArgumentParser) -> None:
    """Add common data source options to a parser."""
    parser.add_argument(
        "--csv",
        required=False,
        default=None,
        help=(
            "Ścieżka do pliku CSV z danymi OHLC (jeśli brak, szuka automatycznie "
            f"w {DEFAULT_DATA_DIR}/<SYMBOL>_H1.csv)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            f"Katalog z danymi OHLC (domyślnie {DEFAULT_DATA_DIR}, można nadpisać "
            "zmienną FOREST5_DATA_DIR)"
        ),
    )
    parser.add_argument("--time-col", default=None, help="Nazwa kolumny czasu (opcjonalnie)")
    parser.add_argument(
        "--sep",
        default=None,
        help="Separator CSV (np. ';'). Brak = autodetekcja",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        choices=ALLOWED_SYMBOLS,
        help=(
            "Symbol (np. EURUSD). Używany do automatycznego wyszukania danych w "
            f"{DEFAULT_DATA_DIR}/<SYMBOL>_H1.csv"
        ),
    )


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
    add_data_source_args(p_bt)
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
    p_bt.add_argument(
        "--min-confluence", type=float, default=1.0, help="Minimalna konfluencja fuzji"
    )

    p_bt.add_argument("--export-equity", default=None, help="Zapisz equity do CSV")
    p_bt.add_argument("--debug-dir", type=Path, default=None, help="Katalog logów debug")
    p_bt.set_defaults(func=cmd_backtest)

        # grid
    p_gr = sub.add_parser(
        "grid", help="Przeszukiwanie parametrów", formatter_class=SafeHelpFormatter
    )
    p_gr.add_argument("--strategy-name", type=_parse_str_list, default=None, help="Strategy name(s)")
    p_gr.add_argument("--timeframe", type=_parse_str_list, default=None, help="Timeframe(s)")
    p_gr.add_argument("--ema-fast", type=parse_span_or_list, default=None)
    p_gr.add_argument("--ema-slow", type=parse_span_or_list, default=None)
    p_gr.add_argument("--rsi-len", type=parse_span_or_list, default=None)
    p_gr.add_argument("--atr-len", type=parse_span_or_list, default=None)
    p_gr.add_argument("--t-sep-atr", type=parse_span_or_list, default=None)
    p_gr.add_argument("--pullback-atr", type=parse_span_or_list, default=None)
    p_gr.add_argument("--entry-buffer-atr", type=parse_span_or_list, default=None)
    p_gr.add_argument("--sl-buffer-atr", type=parse_span_or_list, default=None)
    p_gr.add_argument("--sl-min-atr", type=parse_span_or_list, default=None)
    p_gr.add_argument("--tp-rr", type=parse_span_or_list, default=None)
    p_gr.add_argument("--trailing-atr", type=parse_span_or_list, default=None)
    p_gr.add_argument("--setup-ttl-bars", type=parse_span_or_list, default=None)
    p_gr.add_argument("--engulf-eps-atr", type=parse_span_or_list, default=None)
    p_gr.add_argument("--engulf-body-min", type=parse_span_or_list, default=None)
    p_gr.add_argument("--pinbar-wick-dom", type=parse_span_or_list, default=None)
    p_gr.add_argument("--pinbar-body-max", type=parse_span_or_list, default=None)
    p_gr.add_argument("--pinbar-opp-wick-max", type=parse_span_or_list, default=None)
    p_gr.add_argument("--star-reclaim-min", type=parse_span_or_list, default=None)
    p_gr.add_argument("--star-mid-small-max", type=parse_span_or_list, default=None)
    p_gr.add_argument("--q-low", type=parse_span_or_list, default=None)
    p_gr.add_argument("--q-high", type=parse_span_or_list, default=None)
    p_gr.add_argument("--disable-timeonly", action="store_true")
    p_gr.add_argument("--pairs", type=_parse_str_list, default=None)
    p_gr.add_argument("--from", dest="from_", default=None)
    p_gr.add_argument("--to", default=None)
    p_gr.add_argument("--seed", type=int, default=None)
    p_gr.add_argument("--out", default=None)
    p_gr.add_argument("--parallel", action="store_true")
    p_gr.add_argument("--max-workers", type=int, default=None)
    p_gr.set_defaults(func=cmd_grid)

    # live
    p_lv = sub.add_parser("live", help="Uruchom trading na żywo", formatter_class=SafeHelpFormatter)
    p_lv.add_argument("--config", required=True, help="Ścieżka do pliku YAML/JSON z ustawieniami")
    p_lv.add_argument("--paper", action="store_true", help="Wymuś paper trading (broker.type)")
    p_lv.add_argument("--debug-dir", type=Path, default=None, help="Katalog logów debug")
    p_lv.set_defaults(func=cmd_live)

    return p


def main(argv: list[str] | None = None) -> int:
    setup_logger()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
