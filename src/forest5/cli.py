from __future__ import annotations

import argparse
import os
import re
import sys
import random
from pathlib import Path
from typing import Optional

import numpy as np
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
from forest5.grid.engine import plan_param_grid
from forest5.live.live_runner import run_live
from forest5.utils.io import (
    read_ohlc_csv,
    load_symbol_csv,
    read_ohlc_csv_smart,
    sniff_csv_dialect,
)
from forest5.utils.timeindex import ensure_h1
from forest5.utils.argparse_ext import PercentAction, span_or_list
from forest5.utils.log import (
    setup_logger,
    log_event,
    E_DATA_CSV_SCHEMA,
    E_DATA_TIME_GAPS,
)


# Backwards compatibility – old name used in previous versions/tests
def _parse_range(spec: str) -> list[int]:
    return span_or_list(spec, int)


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
    path: str | Path,
    time_col: Optional[str] = None,
    sep: Optional[str] = None,
    *,
    policy: str = "strict",
) -> tuple[pd.DataFrame, dict]:
    """Read OHLC CSV and ensure a 1H time index.

    Parameters
    ----------
    path, time_col, sep:
        See :func:`read_ohlc_csv`.
    policy:
        Handling of missing bars passed through to
        :func:`~forest5.utils.timeindex.ensure_h1`.

    Returns the loaded DataFrame together with metadata describing any gaps
    detected in the index.
    """

    df = read_ohlc_csv(path, time_col=time_col, sep=sep)
    df, meta = ensure_h1(df, policy=policy)
    return df, meta


# ------------------------------- CLI commands --------------------------------


def cmd_backtest(args: argparse.Namespace) -> int:
    if args.csv is not None:
        csv_path = Path(args.csv)
    else:
        data_dir = get_data_dir(args.data_dir)
        csv_path = data_dir / f"{args.symbol}_H1.csv"

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    df = read_ohlc_csv_smart(csv_path, time_col=args.time_col, sep=args.sep)
    log_event(E_DATA_CSV_SCHEMA, path=str(csv_path), notes=df.attrs.get("notes", []))
    df, meta = ensure_h1(df, policy=args.h1_policy)
    gaps = [
        {"start": g.start.isoformat(), "end": g.end.isoformat(), "missing": g.missing}
        for g in meta.get("gaps", [])
    ]
    log_event(E_DATA_TIME_GAPS, path=str(csv_path), gaps=gaps)
    if args.time_from is not None or args.time_to is not None:
        df = df.loc[args.time_from : args.time_to]

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

    if args.h1_policy == "drop" and settings.setup_ttl_minutes is None:
        step = meta.get("median_bar_minutes")
        if step:
            settings.setup_ttl_minutes = int(settings.setup_ttl_bars * step)

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
    if args.csv:
        csv_path = Path(args.csv)
    else:
        data_dir = get_data_dir(args.data_dir)
        csv_path = data_dir / f"{args.symbol}_H1.csv"

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    df = read_ohlc_csv_smart(csv_path, time_col=args.time_col, sep=args.sep)
    log_event(E_DATA_CSV_SCHEMA, path=str(csv_path), notes=df.attrs.get("notes", []))
    df, meta = ensure_h1(df, policy=args.h1_policy)
    gaps = [
        {"start": g.start.isoformat(), "end": g.end.isoformat(), "missing": g.missing}
        for g in meta.get("gaps", [])
    ]
    log_event(E_DATA_TIME_GAPS, path=str(csv_path), gaps=gaps)
    if args.time_from is not None or args.time_to is not None:
        df = df.loc[args.time_from : args.time_to]

    fast_vals = [int(v) for v in args.fast_values]
    slow_vals = [int(v) for v in args.slow_values]
    risk_vals = [float(v) for v in args.risk_values] if args.risk_values else None
    max_dd_vals = [float(v) for v in args.max_dd_values] if args.max_dd_values else None

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.time_model and not os.path.exists(args.time_model):
        print(f"Plik modelu czasu nie istnieje: {args.time_model}")
        sys.exit(1)

    kwargs = dict(
        symbol=args.symbol,
        fast_values=fast_vals,
        slow_values=slow_vals,
        capital=float(args.capital),
        risk=float(args.risk),
        max_dd=float(args.max_dd),
        fee=float(args.fee),
        slippage=float(args.slippage),
        atr_period=int(args.atr_period[0]),
        atr_multiple=float(args.atr_multiple),
        use_rsi=bool(args.use_rsi),
        rsi_period=int(args.rsi_period[0]),
        rsi_oversold=int(args.rsi_oversold),
        rsi_overbought=int(args.rsi_overbought),
        t_sep_atr=float(args.t_sep_atr[0]),
        pullback_atr=float(args.pullback_atr[0]),
        entry_buffer_atr=float(args.entry_buffer_atr[0]),
        sl_atr=float(args.sl_atr),
        sl_min_atr=float(args.sl_min_atr[0]),
        rr=float(args.rr[0]),
        q_low=float(args.q_low[0]),
        q_high=float(args.q_high[0]),
        time_model=args.time_model,
        min_confluence=float(args.min_confluence),
        n_jobs=int(args.jobs),
    )
    kwargs["strategy"] = args.strategy
    kwargs["patterns"] = {
        "engulf": {"enabled": bool(args.pat_engulf)},
        "pinbar": {"enabled": bool(args.pat_pinbar)},
        "star": {"enabled": bool(args.pat_star)},
    }
    if args.h1_policy == "drop":
        step = meta.get("median_bar_minutes")
        if step:
            kwargs["setup_ttl_minutes"] = int(step)
    if risk_vals:
        kwargs["risk_values"] = risk_vals
    if max_dd_vals:
        kwargs["max_dd_values"] = max_dd_vals
    if len(args.rsi_period) > 1:
        kwargs.pop("rsi_period", None)
        kwargs["rsi_period_values"] = [int(v) for v in args.rsi_period]
    if args.debug_dir:
        kwargs["debug_dir"] = args.debug_dir
    if args.seed is not None:
        kwargs["seed"] = int(args.seed)

    if args.dry_run:
        param_ranges = {"fast": fast_vals, "slow": slow_vals}
        if risk_vals:
            param_ranges["risk"] = risk_vals
        if max_dd_vals:
            param_ranges["max_dd"] = max_dd_vals
        if len(args.rsi_period) > 1:
            param_ranges["rsi_period"] = [int(v) for v in args.rsi_period]
        combos = plan_param_grid(param_ranges, filter_fn=lambda c: c["fast"] < c["slow"])
        combos.to_csv("plan.csv", index=False)
        from datetime import datetime, timezone
        import json
        import subprocess

        git_rev = "unknown"
        try:  # pragma: no cover - best effort
            git_rev = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:  # pragma: no cover
            pass

        meta = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_rev,
            "symbol": args.symbol,
            "period": {
                "start": df.index[0].isoformat(),
                "end": df.index[-1].isoformat(),
            },
            "seed": int(args.seed) if args.seed is not None else None,
            "total_combos": int(len(combos)),
        }
        Path("meta.json").write_text(json.dumps(meta))
        print(len(combos))
        return 0

    out = run_grid(df, **kwargs)

    # sortuj wg RAR / Sharpe jeśli dostępne, inaczej equity_end
    sort_cols = [c for c in ("rar", "sharpe", "equity_end") if c in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, ascending=False)

    head = out.head(args.top)
    print(head)

    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() in (".parquet", ".pq"):
            out.to_parquet(out_path, index=False)
        else:
            out.to_csv(out_path, index=False)
        print(f"Zapisano wyniki grid do: {out_path.resolve()}")

    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    if args.csv:
        df, meta = load_ohlc_csv(
            args.csv,
            time_col=args.time_col,
            sep=args.sep,
            policy=args.h1_policy,
        )
    else:
        df, meta = load_symbol_csv(
            args.symbol, data_dir=args.data_dir, policy=args.h1_policy
        )

    def _single_val(spec: str) -> int:
        vals = span_or_list(spec, int)
        if len(vals) != 1:
            raise ValueError("Expected single value for strategy parameter")
        return vals[0]

    ema_fast = _single_val(args.ema_fast)
    ema_slow = _single_val(args.ema_slow)
    rsi_len = _single_val(args.rsi_len)
    atr_len = _single_val(args.atr_len)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    settings = BacktestSettings(
        symbol=args.symbol,
        timeframe="1h",
        strategy=dict(
            name=args.strategy,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            atr_period=atr_len,
            rsi_period=rsi_len,
            t_sep_atr=args.t_sep_atr,
            pullback_atr=args.pullback_atr,
            entry_buffer_atr=args.entry_buffer_atr,
            sl_atr=args.sl_atr,
            sl_min_atr=args.sl_min_atr,
            rr=args.rr,
            patterns={
                "engulf": {"enabled": bool(args.pat_engulf)},
                "pinbar": {"enabled": bool(args.pat_pinbar)},
                "star": {"enabled": bool(args.pat_star)},
            },
        ),
        risk=dict(
            initial_capital=float(args.capital),
            risk_per_trade=float(args.risk),
            max_drawdown=float(args.max_dd),
            fee_perc=float(args.fee),
            slippage_perc=float(args.slippage),
        ),
        atr_period=atr_len,
        atr_multiple=args.atr_multiple,
        debug_dir=args.debug_dir,
    )

    settings.time.q_low = float(args.q_low)
    settings.time.q_high = float(args.q_high)

    if args.h1_policy == "drop" and settings.setup_ttl_minutes is None:
        step = meta.get("median_bar_minutes")
        if step:
            settings.setup_ttl_minutes = int(settings.setup_ttl_bars * step)

    train = int(args.train)
    test = int(args.test)
    start = 0
    rows: list[dict[str, object]] = []
    base_seed = int(args.seed) if args.seed is not None else None

    while start + train + test <= len(df):
        if args.mode == "anchored":
            train_df = df.iloc[: start + train]
        else:
            train_df = df.iloc[start : start + train]
        test_df = df.iloc[start + train : start + train + test]

        if not args.dry_run:
            if base_seed is not None:
                local_seed = base_seed + len(rows)
                random.seed(local_seed)
                np.random.seed(local_seed)
            res = run_backtest(test_df, settings)
            eq = res.equity_curve
            eq_end = float(eq.iloc[-1]) if not eq.empty else 0.0
            rows.append(
                {
                    "train_start": train_df.index[0],
                    "train_end": train_df.index[-1],
                    "test_start": test_df.index[0],
                    "test_end": test_df.index[-1],
                    "equity_end": eq_end,
                }
            )
        else:
            rows.append(
                {
                    "train_start": train_df.index[0],
                    "train_end": train_df.index[-1],
                    "test_start": test_df.index[0],
                    "test_end": test_df.index[-1],
                }
            )
        start += test

    if args.out:
        out_path = Path(args.out)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Zapisano wyniki WF do: {out_path.resolve()}")

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


def cmd_data_inspect(args: argparse.Namespace) -> int:
    path = Path(args.csv)
    sep, dec, has_header = sniff_csv_dialect(path)
    header = "yes" if has_header else "no"
    print(f"dialect: sep='{sep}' decimal='{dec}' header={header}")
    df = read_ohlc_csv_smart(path, time_col=args.time_col, sep=args.sep, decimal=args.decimal)
    print(f"schema: {', '.join(df.columns)}")
    if df.empty:
        print("no data")
        return 0
    start, end = df.index[0], df.index[-1]
    print(f"date range: {start} -> {end} ({len(df)} rows)")
    _, meta = ensure_h1(df, policy="pad")
    gaps = meta.get("gaps", [])
    if not gaps:
        print("gaps: none")
    else:
        print("gaps preview:")
        for g in gaps[:5]:
            print(f"  {g.start} -> {g.end} (missing {g.missing})")
        if len(gaps) > 5:
            print(f"  ... {len(gaps) - 5} more")
    return 0


def cmd_data_normalize(args: argparse.Namespace) -> int:
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.csv")):
        df = read_ohlc_csv_smart(path)
        out_path = out_dir / path.name
        df.to_csv(out_path, index_label="time")
        print(f"{path.name} -> {out_path}")
    return 0


def cmd_data_pad_h1(args: argparse.Namespace) -> int:
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.csv")):
        df = read_ohlc_csv_smart(path)
        df, _ = ensure_h1(df, policy="pad")
        out_path = out_dir / path.name
        df.to_csv(out_path, index_label="time")
        print(f"{path.name} -> {out_path}")
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
    parser.add_argument(
        "--h1-policy",
        choices=("strict", "pad", "drop"),
        default="strict",
        help=("Jak traktować braki 1H: 'strict' = błąd, 'pad' = wstaw NaN, 'drop' = usuń"),
    )
    parser.add_argument(
        "--from",
        dest="time_from",
        type=pd.Timestamp,
        default=None,
        help="Początek zakresu danych (ISO‑datetime)",
    )
    parser.add_argument(
        "--to",
        dest="time_to",
        type=pd.Timestamp,
        default=None,
        help="Koniec zakresu danych (ISO‑datetime)",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forest5",
        description="Forest 5.0 – modularny framework tradingowy.",
        formatter_class=SafeHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # data utilities
    p_data = sub.add_parser("data", help="Narzędzia danych", formatter_class=SafeHelpFormatter)
    sub_data = p_data.add_subparsers(dest="data_cmd")
    sub_data.required = True

    p_ins = sub_data.add_parser(
        "inspect", help="Pokaż informacje o CSV", formatter_class=SafeHelpFormatter
    )
    p_ins.add_argument("--csv", required=True, help="Plik CSV do analizy")
    p_ins.add_argument("--time-col", default=None, help="Nazwa kolumny czasu")
    p_ins.add_argument("--sep", default=None, help="Separator CSV")
    p_ins.add_argument("--decimal", default=None, help="Separator dziesiętny")
    p_ins.set_defaults(func=cmd_data_inspect)

    p_norm = sub_data.add_parser(
        "normalize", help="Normalizuj pliki CSV", formatter_class=SafeHelpFormatter
    )
    p_norm.add_argument("--input-dir", type=Path, required=True, help="Katalog wejściowy")
    p_norm.add_argument("--out-dir", type=Path, required=True, help="Katalog wyjściowy")
    p_norm.set_defaults(func=cmd_data_normalize)

    p_pad = sub_data.add_parser(
        "pad-h1", help="Uzupełnij braki do 1H", formatter_class=SafeHelpFormatter
    )
    p_pad.add_argument("--input-dir", type=Path, required=True, help="Katalog wejściowy")
    p_pad.add_argument("--out-dir", type=Path, required=True, help="Katalog wyjściowy")
    p_pad.set_defaults(func=cmd_data_pad_h1)

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
    add_data_source_args(p_gr)
    p_gr.add_argument(
        "--ema-fast",
        "--fast-values",
        "--fast",
        dest="fast_values",
        type=span_or_list,
        required=True,
        help="Np. 5:20:1 lub 5,8,13",
    )
    p_gr.add_argument(
        "--ema-slow",
        "--slow-values",
        "--slow",
        dest="slow_values",
        type=span_or_list,
        required=True,
        help="Np. 10:60:2 lub 12,26",
    )
    p_gr.add_argument(
        "--strategy",
        default="h1_ema_rsi_atr",
        help="Nazwa strategii",
    )
    p_gr.add_argument(
        "--risk-values",
        type=span_or_list,
        default=None,
        help="Lista wartości ryzyka, np. 0.01,0.02",
    )
    p_gr.add_argument(
        "--max-dd-values",
        type=span_or_list,
        default=None,
        help="Lista wartości max DD, np. 0.2,0.3",
    )
    p_gr.add_argument("--capital", type=float, default=100_000.0)
    p_gr.add_argument("--risk", action=PercentAction, default=0.01)
    p_gr.add_argument("--max-dd", action=PercentAction, default=0.30, help="Dozwolone obsunięcie")
    p_gr.add_argument("--fee", action=PercentAction, default=0.0005, help="Prowizja %")
    p_gr.add_argument("--slippage", action=PercentAction, default=0.0, help="Poślizg %")

    p_gr.add_argument(
        "--atr-len", "--atr-period", dest="atr_period", type=span_or_list, default=[14]
    )
    p_gr.add_argument("--atr-multiple", type=float, default=2.0)

    p_gr.add_argument("--use-rsi", action="store_true", help="Włącz filtr RSI")
    p_gr.add_argument(
        "--rsi-len", "--rsi-period", dest="rsi_period", type=span_or_list, default=[14]
    )
    p_gr.add_argument("--rsi-oversold", type=int, default=30, choices=range(0, 101))
    p_gr.add_argument("--rsi-overbought", type=int, default=70, choices=range(0, 101))

    p_gr.add_argument("--t-sep-atr", dest="t_sep_atr", type=span_or_list, default=[0.5])
    p_gr.add_argument(
        "--pullback-atr",
        "--pullback-to-ema-fast-atr",
        dest="pullback_atr",
        type=span_or_list,
        default=[0.5],
    )
    p_gr.add_argument(
        "--entry-buffer-atr", dest="entry_buffer_atr", type=span_or_list, default=[0.1]
    )
    p_gr.add_argument("--sl-atr", dest="sl_atr", type=float, default=1.0)
    p_gr.add_argument("--sl-min-atr", dest="sl_min_atr", type=span_or_list, default=[0.0])
    p_gr.add_argument("--rr", dest="rr", type=span_or_list, default=[2.0])
    p_gr.add_argument("--trailing-atr", dest="trailing_atr", type=span_or_list, default=[0.0])
    p_gr.add_argument("--q-low", dest="q_low", type=span_or_list, default=[0.1])
    p_gr.add_argument("--q-high", dest="q_high", type=span_or_list, default=[0.9])
    p_gr.add_argument("--no-engulf", dest="pat_engulf", action="store_false", default=True)
    p_gr.add_argument("--no-pinbar", dest="pat_pinbar", action="store_false", default=True)
    p_gr.add_argument("--no-star", dest="pat_star", action="store_false", default=True)

    p_gr.add_argument("--time-model", type=Path, default=None, help="Ścieżka do modelu czasu")
    p_gr.add_argument(
        "--min-confluence", type=float, default=1.0, help="Minimalna konfluencja fuzji"
    )

    p_gr.add_argument("--dry-run", action="store_true", help="Tylko pokaż konfigurację")
    p_gr.add_argument("--seed", type=int, default=None, help="Losowe ziarno")
    p_gr.add_argument("--jobs", type=int, default=1, help="Równoległość (1 = sekwencyjnie)")
    p_gr.add_argument("--top", type=int, default=20, help="Ile rekordów wyświetlić")
    p_gr.add_argument("--out", "--export", dest="out", default=None, help="Zapis do CSV/Parquet")
    p_gr.add_argument("--debug-dir", type=Path, default=None, help="Katalog logów debug")
    p_gr.set_defaults(func=cmd_grid)

    # walkforward
    p_wf = sub.add_parser(
        "walkforward",
        help="Walk-forward evaluation",
        formatter_class=SafeHelpFormatter,
    )
    add_data_source_args(p_wf)
    p_wf.add_argument("--train", type=int, required=True, help="Okno treningowe (liczba barów)")
    p_wf.add_argument("--test", type=int, required=True, help="Okno testowe (liczba barów)")
    p_wf.add_argument("--mode", choices=("rolling", "anchored"), default="rolling")
    p_wf.add_argument("--strategy", default="h1_ema_rsi_atr")
    p_wf.add_argument("--ema-fast", required=True)
    p_wf.add_argument("--ema-slow", required=True)
    p_wf.add_argument("--rsi-len", default="14")
    p_wf.add_argument("--atr-len", default="14")
    p_wf.add_argument("--t-sep-atr", dest="t_sep_atr", type=float, default=0.5)
    p_wf.add_argument(
        "--pullback-atr",
        "--pullback-to-ema-fast-atr",
        dest="pullback_atr",
        type=float,
        default=0.5,
    )
    p_wf.add_argument("--entry-buffer-atr", dest="entry_buffer_atr", type=float, default=0.1)
    p_wf.add_argument("--sl-atr", dest="sl_atr", type=float, default=1.0)
    p_wf.add_argument("--sl-min-atr", dest="sl_min_atr", type=float, default=0.0)
    p_wf.add_argument("--rr", dest="rr", type=float, default=2.0)
    p_wf.add_argument("--q-low", dest="q_low", type=float, default=0.1)
    p_wf.add_argument("--q-high", dest="q_high", type=float, default=0.9)
    p_wf.add_argument("--no-engulf", dest="pat_engulf", action="store_false", default=True)
    p_wf.add_argument("--no-pinbar", dest="pat_pinbar", action="store_false", default=True)
    p_wf.add_argument("--no-star", dest="pat_star", action="store_false", default=True)
    p_wf.add_argument("--capital", type=float, default=100_000.0)
    p_wf.add_argument("--risk", action=PercentAction, default=0.01)
    p_wf.add_argument("--max-dd", action=PercentAction, default=0.30)
    p_wf.add_argument("--fee", action=PercentAction, default=0.0005)
    p_wf.add_argument("--slippage", action=PercentAction, default=0.0)
    p_wf.add_argument("--atr-multiple", type=float, default=2.0)
    p_wf.add_argument("--dry-run", action="store_true")
    p_wf.add_argument("--seed", type=int, default=None)
    p_wf.add_argument("--jobs", type=int, default=1)
    p_wf.add_argument("--out", default=None)
    p_wf.add_argument("--debug-dir", type=Path, default=None)
    p_wf.set_defaults(func=cmd_walkforward)

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
