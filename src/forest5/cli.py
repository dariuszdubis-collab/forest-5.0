from __future__ import annotations

import argparse
import os
import re
import sys
import random
import json
import math
import time
import socket
import subprocess  # nosec B404
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

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
from forest5.backtest.grid import make_combo_id
from forest5.grid.engine import plan_param_grid, run_grid
from forest5.live.live_runner import run_live
from forest5.utils.io import (
    read_ohlc_csv,
    load_symbol_csv,
    read_ohlc_csv_smart,
    sniff_csv_dialect,
    atomic_to_csv,
    atomic_write_json,
    normalize_ohlc_h1,
)
from forest5.utils.timeindex import ensure_h1
from forest5.utils.argparse_ext import (
    PercentAction,
    span_or_list,
    SpanOrList,
    positive_int,
    validate_chunks,
)
from forest5.utils.log import (
    setup_logger,
    log_event,
    E_PREFLIGHT_ACK,
)
from forest5.live.mt4_broker import MT4Broker


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
    df = normalize_ohlc_h1(df, policy=policy)
    return df, meta


PATTERN_PARAM_KEYS = [
    "engulf_eps_atr",
    "engulf_body_ratio_min",
    "pinbar_wick_dom",
    "pinbar_body_max",
    "pinbar_opp_wick_max",
    "star_reclaim_min",
    "star_mid_small_max",
]


def _collect_pattern_overrides_ns(ns: argparse.Namespace) -> Dict[str, Any]:
    """Extract pattern-related overrides from a namespace."""

    d: Dict[str, Any] = {}
    for k in PATTERN_PARAM_KEYS:
        v = getattr(ns, k, None)
        if v is not None:
            d[k] = v
    if getattr(ns, "no_engulf", False):
        d["enable_engulf"] = False
    if getattr(ns, "no_pinbar", False):
        d["enable_pinbar"] = False
    if getattr(ns, "no_star", False):
        d["enable_star"] = False
    return d


# ------------------------------- CLI commands --------------------------------


def cmd_validate_live_config(args: argparse.Namespace) -> int:
    from forest5.config_live import validate_live_config

    ok, details = validate_live_config(args.yaml, strict=args.strict)
    if ok:
        print(details.get("message", "OK"))
        return 0
    print(details.get("error", "Invalid config"), file=sys.stderr)
    return 2


def add_validate_subparser(sub: argparse._SubParsersAction) -> None:
    """Attach ``validate`` subcommands to the top level parser."""

    p = sub.add_parser("validate", help="validation utilities")
    sp = p.add_subparsers(dest="validate_cmd")
    sp.required = True

    v = sp.add_parser("live-config", help="validate live trading config")
    v.add_argument("--yaml", required=True, help="ścieżka do pliku YAML")
    v.add_argument("--strict", action="store_true", help="zakończ błędem przy ostrzeżeniach")
    v.set_defaults(func=cmd_validate_live_config)


def cmd_backtest(args: argparse.Namespace) -> int:
    if args.csv is not None:
        csv_path = Path(args.csv)
    else:
        data_dir = get_data_dir(args.data_dir)
        csv_path = data_dir / f"{args.symbol}_H1.csv"

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    df = read_ohlc_csv(csv_path, time_col=args.time_col, sep=args.sep)
    df, time_meta = ensure_h1(df, policy=args.h1_policy)
    df = normalize_ohlc_h1(df, policy=args.h1_policy)
    if args.time_from is not None or args.time_to is not None:
        df = df.loc[args.time_from : args.time_to]

    if args.strategy == "ema_cross":
        strat_cfg: Dict[str, Any] = {
            "name": "ema_cross",
            "fast": args.fast,
            "slow": args.slow,
            "use_rsi": bool(args.use_rsi),
            "rsi_period": args.rsi_period,
            "rsi_oversold": args.rsi_oversold,
            "rsi_overbought": args.rsi_overbought,
        }
    else:
        strat_cfg = {
            "name": "h1_ema_rsi_atr",
            "ema_fast": args.ema_fast,
            "ema_slow": args.ema_slow,
            "rsi_period": args.rsi_len,
            "atr_period": args.atr_len,
            "t_sep_atr": args.t_sep_atr,
            "pullback_atr": args.pullback_atr,
            "entry_buffer_atr": args.entry_buffer_atr,
            "sl_min_atr": args.sl_min_atr,
            "rr": args.rr,
        }

        overrides = _collect_pattern_overrides_ns(args)
        if overrides:
            strat_cfg.update(overrides)

    settings = BacktestSettings(
        symbol=args.symbol,
        timeframe="1h",
        strategy=strat_cfg,
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
        step = time_meta.get("median_bar_minutes")
        if step:
            settings.setup_ttl_minutes = int(settings.setup_ttl_bars * step)

    settings.time.model.enabled = bool(args.time_model)
    settings.time.model.path = args.time_model
    settings.time.fusion_min_confluence = float(args.min_confluence)
    settings.time.q_low = float(args.q_low)
    settings.time.q_high = float(args.q_high)

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
    try:
        validate_chunks(args.chunks, args.chunk_id)
    except argparse.ArgumentTypeError as exc:  # pragma: no cover - argparse style
        print(str(exc), file=sys.stderr)
        return 2

    if args.csv:
        csv_path = Path(args.csv)
    else:
        data_dir = get_data_dir(args.data_dir)
        csv_path = data_dir / f"{args.symbol}_H1.csv"

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    df = read_ohlc_csv(csv_path, time_col=args.time_col, sep=args.sep)
    df, time_meta = ensure_h1(df, policy=args.h1_policy)
    df = normalize_ohlc_h1(df, policy=args.h1_policy)
    if args.time_from is not None or args.time_to is not None:
        df = df.loc[args.time_from : args.time_to]

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.time_model and not os.path.exists(args.time_model):
        print(f"Plik modelu czasu nie istnieje: {args.time_model}")
        sys.exit(1)

    fast_vals = [int(v) for v in args.fast_values]
    slow_vals = [int(v) for v in args.slow_values]
    param_ranges: Dict[str, list[Any]] = {"fast": fast_vals, "slow": slow_vals}
    if args.risk_values:
        param_ranges["risk"] = [float(v) for v in args.risk_values]
    if args.max_dd_values:
        param_ranges["max_dd"] = [float(v) for v in args.max_dd_values]
    if len(args.rsi_period) > 1:
        param_ranges["rsi_period"] = [int(v) for v in args.rsi_period]

    pattern_ranges = _collect_pattern_overrides_ns(args)
    base_pattern_vals: Dict[str, Any] = {}
    for k, v in pattern_ranges.items():
        if isinstance(v, list):
            param_ranges[k] = list(v)
            base_pattern_vals[k] = v[0]
        else:
            param_ranges[k] = [v]
            base_pattern_vals[k] = v

    combos_all = plan_param_grid(param_ranges, filter_fn=lambda c: c["fast"] < c["slow"])
    if args.out:
        out_dir = Path(args.out)
    else:
        # default to a directory next to the CSV to avoid sharing state across tests
        out_dir = Path(csv_path).resolve().parent / "out"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.csv"
    results_path = Path(args.results) if getattr(args, "results", None) else out_dir / "results.csv"
    meta_path = Path(args.meta_out) if getattr(args, "meta_out", None) else out_dir / "meta.json"
    top_path = out_dir / "results_top.csv"

    atomic_to_csv(combos_all, plan_path)

    done_df = None
    done_ids: set[str] = set()
    resume_mode = args.resume
    if resume_mode in ("auto", "on") and results_path.exists():
        done_df = pd.read_csv(results_path)
        param_names = list(param_ranges.keys())
        if "combo_id" not in done_df.columns:
            done_df["combo_json"] = [
                json.dumps(
                    {k: row.get(k) for k in param_names if k in row},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for row in done_df.to_dict("records")
            ]
            done_df["combo_id"] = [make_combo_id(json.loads(js)) for js in done_df["combo_json"]]
        elif "combo_json" not in done_df.columns:
            done_df["combo_json"] = [
                json.dumps(
                    {k: row.get(k) for k in param_names if k in row},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for row in done_df.to_dict("records")
            ]
        done_ids = set(done_df["combo_id"].astype(str))
        print(
            f"[resume] skipping {len(done_ids)} of {len(combos_all)} combos already in results.csv"
        )

    combos = combos_all
    if args.chunks is not None and args.chunk_id is not None:
        size = math.ceil(len(combos_all) / args.chunks)
        start = (args.chunk_id - 1) * size
        end = args.chunk_id * size
        combos = combos_all.iloc[start:end]
        if args.dry_run:
            combos.to_csv(
                out_dir / f"plan_shard_{args.chunk_id:02d}of{args.chunks:02d}.csv",
                index=False,
            )

    if done_ids:
        combos = combos[~combos["combo_id"].astype(str).isin(done_ids)]
        if combos.empty:
            print("nothing to do (resume)")
            return 0

    if args.dry_run:
        run_meta = {
            "symbol": args.symbol,
            "seed": args.seed,
            "total_combos": int(len(combos_all)),
        }
        atomic_write_json(run_meta, meta_path)
        print(len(combos_all))
        return 0

    jobs = args.jobs
    if jobs is None:
        env_jobs = os.environ.get("FOREST5_TEST_JOBS")
        if env_jobs is not None:
            jobs = int(env_jobs)
        elif os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true":
            jobs = 0
        else:
            jobs = 1
    jobs = int(jobs)

    start_time = datetime.now(timezone.utc)
    git_sha = "unknown"
    try:  # pragma: no cover - best effort
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover
        pass

    run_meta = {
        "iso_datetime_start": start_time.isoformat(),
        "command_line": " ".join(sys.argv),
        "seed": args.seed,
        "jobs": jobs,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "git_sha": git_sha,
        "input_csv": str(csv_path),
        "symbol": args.symbol,
        "strategy": args.strategy,
        "plan_total": int(len(combos_all)),
        "plan_remaining": int(len(combos)),
    }
    atomic_write_json(run_meta, meta_path)

    settings = BacktestSettings(
        symbol=args.symbol,
        timeframe="1h",
        strategy=dict(
            name=args.strategy,
            fast=fast_vals[0],
            slow=slow_vals[0],
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
            **base_pattern_vals,
        ),
        risk=dict(
            initial_capital=float(args.capital),
            risk_per_trade=float(args.risk),
            max_drawdown=float(args.max_dd),
            fee_perc=float(args.fee),
            slippage_perc=float(args.slippage),
        ),
        atr_period=int(args.atr_period[0]),
        atr_multiple=float(args.atr_multiple),
        debug_dir=args.debug_dir,
    )
    settings.time.q_low = float(args.q_low[0])
    settings.time.q_high = float(args.q_high[0])
    settings.time.model.enabled = bool(args.time_model)
    settings.time.model.path = args.time_model
    settings.time.fusion_min_confluence = float(args.min_confluence)
    if args.h1_policy == "drop" and settings.setup_ttl_minutes is None:
        step = time_meta.get("median_bar_minutes")
        if step:
            settings.setup_ttl_minutes = int(settings.setup_ttl_bars * step)

    new_results = run_grid(df, combos, settings, jobs=jobs, seed=args.seed)

    if done_df is not None:
        merged = pd.concat([done_df, new_results], ignore_index=True)
        merged = merged.drop_duplicates(subset=["combo_id"], keep="last")
    else:
        merged = new_results

    sort_cols: list[str] = []
    sort_asc: list[bool] = []
    if "equity_end" in merged.columns:
        sort_cols.append("equity_end")
        sort_asc.append(False)
    if "trades" in merged.columns:
        sort_cols.append("trades")
        sort_asc.append(False)
    if "dd" in merged.columns:
        sort_cols.append("dd")
        sort_asc.append(True)
    elif "max_dd" in merged.columns:
        sort_cols.append("max_dd")
        sort_asc.append(True)
    if sort_cols:
        merged = merged.sort_values(by=sort_cols, ascending=sort_asc)

    atomic_to_csv(merged, results_path)

    topn = int(args.top)
    top_df = merged.head(topn)
    atomic_to_csv(top_df, top_path)
    print("\n=== TOP", topn, "===\n", top_df.to_string(index=False))

    end_time = datetime.now(timezone.utc)
    success = int((merged.get("error").isna()).sum()) if "error" in merged.columns else len(merged)
    failed = int((merged.get("error").notna()).sum()) if "error" in merged.columns else 0
    run_meta.update(
        {
            "iso_datetime_end": end_time.isoformat(),
            "duration_sec": (end_time - start_time).total_seconds(),
            "success": success,
            "failed": failed,
            "aborted": 0,
            "completed_combos": int(len(merged)),
        }
    )
    atomic_write_json(run_meta, meta_path)
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
        df, meta = load_symbol_csv(args.symbol, data_dir=args.data_dir, policy=args.h1_policy)

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
                "engulfing": args.pat_engulf,
                "pinbar": args.pat_pinbar,
                "stars": args.pat_star,
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
    if args.config is None:
        print("--config is required", file=sys.stderr)
        return 2
    settings = load_live_settings(args.config)
    if args.paper:
        settings.broker.type = "paper"
    kwargs = {}
    if args.debug_dir:
        kwargs["debug_dir"] = args.debug_dir
    run_live(settings, **kwargs)
    return 0


def cmd_live_preflight(args: argparse.Namespace) -> int:
    """Perform MT4 bridge preflight handshake and save symbol specs."""

    broker = MT4Broker(bridge_dir=args.bridge_dir, symbol=args.symbol, timeout_sec=args.timeout)
    broker.connect()
    uid = broker.request_specs()

    try:
        specs = broker.await_ack(uid, timeout=args.timeout)
    except TimeoutError:
        print("No acknowledgement from MT4 bridge (timeout)", file=sys.stderr)
        return 1

    try:
        specs = broker.validate_specs(specs)
    except ValueError as exc:  # pragma: no cover - defensive
        print(str(exc), file=sys.stderr)
        return 1

    out_path = Path(args.bridge_dir) / "symbol_specs.json"
    out_path.write_text(json.dumps(specs), encoding="utf-8")
    ack_path = Path(args.bridge_dir) / "handshake_ack.json"
    payload = {"symbol": args.symbol, "timestamp": time.time()}
    ack_path.write_text(json.dumps(payload), encoding="utf-8")
    log_event(E_PREFLIGHT_ACK, symbol=args.symbol, path=str(ack_path))
    # simple table-like output for UX
    print("Symbol specifications:")
    for k, v in specs.items():
        print(f"  {k}: {v}")
    return 0


def _inspect_file(path: Path, args: argparse.Namespace) -> tuple[str, dict]:
    sep, dec, has_header = sniff_csv_dialect(path)
    header = "yes" if has_header else "no"
    lines = [f"dialect: sep='{sep}' decimal='{dec}' header={header}"]
    df = read_ohlc_csv_smart(path, time_col=args.time_col, sep=args.sep, decimal=args.decimal)
    lines.append(f"schema: {', '.join(df.columns)}")
    summary = {
        "dialect": {"sep": sep, "decimal": dec, "header": has_header},
        "schema": list(df.columns),
        "rows": int(len(df)),
        "gaps": [],
    }
    if df.empty:
        lines.append("no data")
        text = "\n".join(lines)
        return text, summary
    start, end = df.index[0], df.index[-1]
    lines.append(f"date range: {start} -> {end} ({len(df)} rows)")
    summary["date_range"] = {"start": start.isoformat(), "end": end.isoformat()}
    _, meta = ensure_h1(df, policy="pad")
    gaps = meta.get("gaps", [])
    if not gaps:
        lines.append("gaps: none")
    else:
        lines.append("gaps preview:")
        preview = []
        for g in gaps[:5]:
            lines.append(f"  {g.start} -> {g.end} (missing {g.missing})")
            preview.append(
                {"start": g.start.isoformat(), "end": g.end.isoformat(), "missing": g.missing}
            )
        summary["gaps"] = preview
        if len(gaps) > 5:
            lines.append(f"  ... {len(gaps) - 5} more")
    text = "\n".join(lines)
    return text, summary


def cmd_data_inspect(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    if args.csv:
        paths = [Path(args.csv)]
    elif args.input_dir:
        paths = sorted(Path(args.input_dir).glob("*.csv"))
    out_dir = Path(args.out) if getattr(args, "out", None) else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        text, summary = _inspect_file(p, args)
        print(text)
        if out_dir:
            (out_dir / "summary.txt").write_text(text, encoding="utf-8")
            (out_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
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
    if args.csv:
        path = Path(args.csv)
        if args.out is None:
            print("--out is required when using --csv", file=sys.stderr)
            return 2
        df = read_ohlc_csv_smart(path)
        try:
            df, _ = ensure_h1(df, policy=args.policy)
            df = normalize_ohlc_h1(df, policy=args.policy)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index_label="time")
        print(f"{path.name} -> {out_path}")
        return 0

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for path in sorted(in_dir.glob("*.csv")):
        df = read_ohlc_csv_smart(path)
        try:
            df, _ = ensure_h1(df, policy=args.policy)
            df = normalize_ohlc_h1(df, policy=args.policy)
        except ValueError as exc:
            print(f"{path.name}: {exc}", file=sys.stderr)
            rc = 1
            continue
        out_path = out_dir / path.name
        df.to_csv(out_path, index_label="time")
        print(f"{path.name} -> {out_path}")
    return rc


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
        epilog=(
            "Performance notes: CSV loader uses float32 and memory mapping, "
            "grid precomputes indicators once, and --jobs controls worker "
            "processes."
        ),
        formatter_class=SafeHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    add_validate_subparser(sub)

    # data utilities
    p_data = sub.add_parser("data", help="Narzędzia danych", formatter_class=SafeHelpFormatter)
    sub_data = p_data.add_subparsers(dest="data_cmd")
    sub_data.required = True

    p_ins = sub_data.add_parser(
        "inspect", help="Pokaż informacje o CSV", formatter_class=SafeHelpFormatter
    )
    grp_ins = p_ins.add_mutually_exclusive_group(required=True)
    grp_ins.add_argument("--csv", help="Plik CSV do analizy")
    grp_ins.add_argument("--input-dir", type=Path, help="Katalog z plikami CSV")
    p_ins.add_argument("--out", type=Path, help="Katalog wynikowy na podsumowanie")
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
    grp_pad = p_pad.add_mutually_exclusive_group(required=True)
    grp_pad.add_argument("--input-dir", type=Path, help="Katalog wejściowy")
    grp_pad.add_argument("--csv", help="Pojedynczy plik CSV")
    p_pad.add_argument("--out-dir", type=Path, help="Katalog wyjściowy (dla --input-dir)")
    p_pad.add_argument("--out", type=Path, help="Plik wyjściowy (dla --csv)")
    p_pad.add_argument(
        "--policy", choices=("strict", "pad"), default="pad", help="Polityka braków czasu"
    )
    p_pad.set_defaults(func=cmd_data_pad_h1)

    # backtest
    p_bt = sub.add_parser(
        "backtest", help="Uruchom pojedynczy backtest", formatter_class=SafeHelpFormatter
    )
    add_data_source_args(p_bt)
    p_bt.add_argument(
        "--strategy",
        choices=("ema_cross", "h1_ema_rsi_atr"),
        default="ema_cross",
        help="Wybór strategii",
    )

    # opcje dla strategii ema_cross
    p_bt.add_argument("--fast", type=int, default=12, help="Szybka EMA")
    p_bt.add_argument("--slow", type=int, default=26, help="Wolna EMA")

    p_bt.add_argument("--use-rsi", action="store_true", help="Włącz filtr RSI")
    p_bt.add_argument("--rsi-period", type=int, default=14)
    p_bt.add_argument("--rsi-oversold", type=int, default=30, choices=range(0, 101))
    p_bt.add_argument("--rsi-overbought", type=int, default=70, choices=range(0, 101))

    # parametry strategii h1_ema_rsi_atr
    p_bt.add_argument("--ema-fast", type=int, default=21)
    p_bt.add_argument("--ema-slow", type=int, default=55)
    p_bt.add_argument("--rsi-len", dest="rsi_len", type=int, default=14)
    p_bt.add_argument("--atr-len", dest="atr_len", type=int, default=14)
    p_bt.add_argument("--t-sep-atr", dest="t_sep_atr", type=float, default=0.5)
    p_bt.add_argument(
        "--pullback-atr",
        "--pullback-to-ema-fast-atr",
        dest="pullback_atr",
        type=float,
        default=0.5,
    )
    p_bt.add_argument(
        "--entry-buffer-atr",
        dest="entry_buffer_atr",
        type=float,
        default=0.1,
    )
    p_bt.add_argument("--sl-min-atr", dest="sl_min_atr", type=float, default=0.0)
    p_bt.add_argument("--rr", dest="rr", type=float, default=2.0)
    p_bt.add_argument("--q-low", dest="q_low", type=float, default=0.1)
    p_bt.add_argument("--q-high", dest="q_high", type=float, default=0.9)

    p_bt.add_argument("--engulf-eps-atr", dest="engulf_eps_atr", type=float, default=None)
    p_bt.add_argument(
        "--engulf-body-ratio-min", dest="engulf_body_ratio_min", type=float, default=None
    )
    p_bt.add_argument("--pinbar-wick-dom", dest="pinbar_wick_dom", type=float, default=None)
    p_bt.add_argument("--pinbar-body-max", dest="pinbar_body_max", type=float, default=None)
    p_bt.add_argument("--pinbar-opp-wick-max", dest="pinbar_opp_wick_max", type=float, default=None)
    p_bt.add_argument("--star-reclaim-min", dest="star_reclaim_min", type=float, default=None)
    p_bt.add_argument("--star-mid-small-max", dest="star_mid_small_max", type=float, default=None)

    p_bt.add_argument("--no-engulf", action="store_true", help="Wyłącz detektor engulfing")
    p_bt.add_argument("--no-pinbar", action="store_true", help="Wyłącz detektor pinbar")
    p_bt.add_argument("--no-star", action="store_true", help="Wyłącz detektor gwiazdy")

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
    p_gr.add_argument(
        "--engulf-eps-atr",
        dest="engulf_eps_atr",
        type=SpanOrList(float),
        default=None,
        help="(span/list) tolerancja domknięcia engulfingu w ATR",
    )
    p_gr.add_argument(
        "--engulf-body-ratio-min",
        dest="engulf_body_ratio_min",
        type=SpanOrList(float),
        default=None,
        help="(span/list) minimalny stosunek ciał świec dla engulfingu",
    )
    p_gr.add_argument(
        "--pinbar-wick-dom",
        dest="pinbar_wick_dom",
        type=SpanOrList(float),
        default=None,
        help="(span/list) dominacja knota pinbara",
    )
    p_gr.add_argument(
        "--pinbar-body-max",
        dest="pinbar_body_max",
        type=SpanOrList(float),
        default=None,
        help="(span/list) maksymalny udział ciała pinbara",
    )
    p_gr.add_argument(
        "--pinbar-opp-wick-max",
        dest="pinbar_opp_wick_max",
        type=SpanOrList(float),
        default=None,
        help="(span/list) maksymalny przeciwny knot pinbara",
    )
    p_gr.add_argument(
        "--star-reclaim-min",
        dest="star_reclaim_min",
        type=SpanOrList(float),
        default=None,
        help="(span/list) minimalny reclaim w formacji Star",
    )
    p_gr.add_argument(
        "--star-mid-small-max",
        dest="star_mid_small_max",
        type=SpanOrList(float),
        default=None,
        help="(span/list) maksymalny rozmiar świecy środkowej w Star",
    )
    p_gr.add_argument(
        "--no-engulf", dest="no_engulf", action="store_true", help="Wyłącz detektor engulfing"
    )
    p_gr.add_argument(
        "--no-pinbar", dest="no_pinbar", action="store_true", help="Wyłącz detektor pinbar"
    )
    p_gr.add_argument(
        "--no-star", dest="no_star", action="store_true", help="Wyłącz detektor gwiazdy"
    )

    p_gr.add_argument("--time-model", type=Path, default=None, help="Ścieżka do modelu czasu")
    p_gr.add_argument(
        "--min-confluence", type=float, default=1.0, help="Minimalna konfluencja fuzji"
    )

    p_gr.add_argument(
        "--resume",
        choices=("auto", "on", "off"),
        default="auto",
        help="Wznów poprzedni bieg (auto wykrywa, on wymusza, off ignoruje)",
    )
    p_gr.add_argument(
        "--chunks",
        type=positive_int,
        default=None,
        help="Podziel siatkę parametrów na N części",
    )
    p_gr.add_argument(
        "--chunk-id",
        type=positive_int,
        default=None,
        help="Uruchom tylko wybrany fragment siatki (1-indexed)",
    )

    p_gr.add_argument("--dry-run", action="store_true", help="Tylko pokaż konfigurację")
    p_gr.add_argument("--seed", type=int, default=None, help="Losowe ziarno")
    p_gr.add_argument("--jobs", type=int, default=None, help="Równoległość (0 = sekwencyjnie)")
    p_gr.add_argument("--top", type=int, default=10, help="Ile rekordów wyeksportować")
    p_gr.add_argument("--results", dest="results", default=None, help="Ścieżka do results.csv")
    p_gr.add_argument("--meta-out", dest="meta_out", default=None, help="Ścieżka do meta.json")
    p_gr.add_argument("--out", "--export", dest="out", default=None, help="Katalog wyjściowy")
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

    # live ----------------------------------------------------------------
    p_lv = sub.add_parser(
        "live",
        help="Uruchom trading na żywo lub wykonaj operacje pomocnicze",
        formatter_class=SafeHelpFormatter,
    )
    sub_lv = p_lv.add_subparsers(dest="live_cmd")
    sub_lv.required = False

    def _add_live_run_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
        parser.add_argument(
            "--config",
            required=required,
            help="Ścieżka do pliku YAML/JSON z ustawieniami",
        )
        parser.add_argument(
            "--paper", action="store_true", help="Wymuś paper trading (broker.type)"
        )
        parser.add_argument("--debug-dir", type=Path, default=None, help="Katalog logów debug")
        parser.set_defaults(func=cmd_live)

    # default run parser so that `forest5 live --config` still works
    _add_live_run_args(p_lv, required=False)
    p_lv.set_defaults(live_cmd="run")

    # explicit `live run` subcommand
    p_lv_run = sub_lv.add_parser(
        "run", help="Uruchom trading na żywo", formatter_class=SafeHelpFormatter
    )
    _add_live_run_args(p_lv_run, required=True)

    # `live preflight` subcommand
    p_lv_pre = sub_lv.add_parser(
        "preflight",
        help="Wykonaj handshake z mostem MT4 i odczytaj specyfikację symbolu",
        formatter_class=SafeHelpFormatter,
    )
    p_lv_pre.add_argument("--bridge-dir", type=Path, required=True, help="Katalog mostu MT4")
    p_lv_pre.add_argument("--symbol", required=True, choices=ALLOWED_SYMBOLS)
    p_lv_pre.add_argument("--timeout", type=float, default=5.0)
    p_lv_pre.set_defaults(func=cmd_live_preflight)

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
