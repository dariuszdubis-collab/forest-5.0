#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_csv_list(spec: str, typ):
    return [typ(x.strip()) for x in spec.split(",") if x.strip()]


def _combo_id(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


@dataclass
class Combo:
    scenario: str
    strategy: str
    ema_fast: int
    ema_slow: int
    rsi_len: int
    atr_len: int
    t_sep_atr: float
    pullback_atr: float
    csv: str
    symbol: str
    h1_policy: str
    time_model: str | None
    q_low: float
    q_high: float

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def id(self) -> str:
        return _combo_id(self.as_dict())


def make_cli_args(c: Combo) -> List[str]:
    args: List[str] = [
        sys.executable,
        "-m",
        "forest5.cli",
        "backtest",
        "--csv",
        c.csv,
        "--symbol",
        c.symbol,
        "--h1-policy",
        c.h1_policy,
        "--q-low",
        str(c.q_low),
        "--q-high",
        str(c.q_high),
    ]

    # Strategy mapping
    if c.strategy == "ema_cross":
        args += ["--strategy", "ema_cross", "--fast", str(c.ema_fast), "--slow", str(c.ema_slow)]
    else:
        args += [
            "--strategy",
            "h1_ema_rsi_atr",
            "--ema-fast",
            str(c.ema_fast),
            "--ema-slow",
            str(c.ema_slow),
            "--rsi-len",
            str(c.rsi_len),
            "--atr-len",
            str(c.atr_len),
            "--t-sep-atr",
            str(c.t_sep_atr),
            "--pullback-atr",
            str(c.pullback_atr),
        ]

    # Scenario flags
    if c.scenario in {"ema_rsi", "rsi_timeonly"}:  # RSI only direction, ATR gates off
        # Disable EMA gates in H1 strategy
        if c.strategy != "ema_cross":
            args += ["--no-ema-gates"]
    if c.scenario in {"atr_timeonly", "rsi_timeonly", "rsi_atr_timeonly"}:
        if c.time_model:
            args += ["--time-model", c.time_model]

    return args


def run_combo(c: Combo, *, dry: bool = False) -> Tuple[str, Dict[str, Any]]:
    combo_id = c.id()
    payload = c.as_dict() | {"combo_id": combo_id}
    if dry:
        return combo_id, {**payload, "dry_run": True}

    # Use metrics-out temp file for machine-readable output
    with tempfile.TemporaryDirectory() as td:
        metrics_path = Path(td) / "metrics.json"
        args = make_cli_args(c) + ["--metrics-out", str(metrics_path)]
        t0 = time.monotonic()
        try:
            project_root = Path(__file__).resolve().parents[1]
            env = os.environ.copy()
            src_path = str(project_root / "src")
            env["PYTHONPATH"] = (
                f"{src_path}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
            )
            proc = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                env=env,
            )
            dur = time.monotonic() - t0
            metrics: Dict[str, Any] = {}
            error: str | None = None
            if proc.returncode != 0:
                error = f"exit={proc.returncode}"
            elif not metrics_path.exists():
                error = "metrics_missing"
            else:
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception as exc:  # pragma: no cover - defensive
                    error = f"metrics_parse: {exc}"  # type: ignore[str-bytes-safe]

            result = {**payload, **metrics, "error": error, "stdout": proc.stdout.strip()}
            status = "OK" if error is None else "ERR"
            trades = result.get("trades")
            eq = result.get("equity_end")
            msg = f"[{status}] {combo_id} {json.dumps({'scenario': c.scenario, 'ema_fast': c.ema_fast, 'ema_slow': c.ema_slow, 'rsi_len': c.rsi_len, 'atr_len': c.atr_len, 't_sep_atr': c.t_sep_atr, 'pullback_atr': c.pullback_atr})} -> trades={trades}, equity={eq} ({dur:.1f}s)"
            print(msg, flush=True)
            if status == "ERR":
                # Also show a compact stderr first line for quick diagnosis
                first_err = (proc.stderr or "").splitlines()[:1]
                if first_err:
                    print(f"       stderr: {first_err[0]}", flush=True)
            return combo_id, result
        except Exception as exc:  # pragma: no cover - defensive
            dur = time.monotonic() - t0
            result = {**payload, "error": f"exception: {exc}"}
            print(
                f"[ERR] {combo_id} (exception) -> {exc!s} ({dur:.1f}s)",
                flush=True,
            )
            return combo_id, result


def build_combos(args: argparse.Namespace) -> List[Combo]:
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    # Parameter lists
    ema_fast_vals = _parse_csv_list(args.ema_fast, int)
    ema_slow_vals = _parse_csv_list(args.ema_slow, int)
    rsi_vals = _parse_csv_list(args.rsi_len, int)
    atr_vals = _parse_csv_list(args.atr_len, int)
    t_sep_vals = _parse_csv_list(args.t_sep_atr, float)
    pullback_vals = _parse_csv_list(args.pullback_atr, float)

    combos: List[Combo] = []
    for sc in scenarios:
        # Map scenario to strategy choice where applicable
        # Default strategy: H1 contract engine
        strategy = "h1_ema_rsi_atr"
        if sc == "ema_atr":  # approximate RSI-OFF via ema_cross (no RSI filter)
            strategy = "ema_cross"

        for f in ema_fast_vals:
            for s in ema_slow_vals:
                for r in rsi_vals:
                    for a in atr_vals:
                        for t in t_sep_vals:
                            for pb in pullback_vals:
                                combos.append(
                                    Combo(
                                        scenario=sc,
                                        strategy=strategy,
                                        ema_fast=int(f),
                                        ema_slow=int(s),
                                        rsi_len=int(r),
                                        atr_len=int(a),
                                        t_sep_atr=float(t),
                                        pullback_atr=float(pb),
                                        csv=args.csv,
                                        symbol=args.symbol,
                                        h1_policy=args.h1_policy,
                                        time_model=args.time_model,
                                        q_low=float(args.q_low),
                                        q_high=float(args.q_high),
                                    )
                                )
    return combos


def main() -> int:
    ap = argparse.ArgumentParser("ablation_scenarios")
    ap.add_argument("--csv", required=True, help="Ścieżka do CSV (H1)")
    ap.add_argument("--symbol", required=True, help="Symbol, np. EURUSD")
    ap.add_argument(
        "--h1-policy",
        choices=("strict", "pad"),
        default="pad",
        help="Polityka braków czasu (do ensure_h1)",
    )
    ap.add_argument(
        "--scenarios",
        default="ema_rsi,ema_atr,atr_timeonly,rsi_timeonly,rsi_atr_timeonly,full",
        help="Lista scenariuszy do uruchomienia (CSV)",
    )
    # parameter lists
    ap.add_argument("--ema-fast", default="12,21")
    ap.add_argument("--ema-slow", default="34,55")
    ap.add_argument("--rsi-len", default="14")
    ap.add_argument("--atr-len", default="14")
    ap.add_argument("--t-sep-atr", default="0.2,0.5")
    ap.add_argument("--pullback-atr", default="0.5,1.0")

    # time-only model
    ap.add_argument("--time-model", default=None, help="Ścieżka do modelu czasu (opcjonalnie)")
    ap.add_argument("--q-low", default=0.1)
    ap.add_argument("--q-high", default=0.9)

    # execution
    ap.add_argument("--max-proc", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--resume", action="store_true", help="Wznów (pomiń wykonane combo_id)")
    ap.add_argument("--limit", type=int, default=None, help="Ogranicz liczbę kombinacji")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("repos/forest-5.0/out_runs/ablation_results.csv"),
        help="Plik wyjściowy CSV",
    )

    args = ap.parse_args()

    out_path: Path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combos = build_combos(args)
    if args.limit is not None:
        combos = combos[: int(args.limit)]

    # Resume mode: load existing combo_ids
    done: set[str] = set()
    if args.resume and out_path.exists():
        try:
            import csv

            with out_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    cid = str(row.get("combo_id", "")).strip()
                    if cid:
                        done.add(cid)
        except Exception:
            done = set()

    # Prepare tasks
    tasks: List[Combo] = []
    for c in combos:
        if args.resume and c.id() in done:
            print(f"[SKIP] {c.id()} already present", flush=True)
            continue
        tasks.append(c)

    if args.dry_run:
        for c in tasks:
            print(f"[PLAN] {c.id()} {json.dumps(c.as_dict(), ensure_ascii=False)}")
        return 0

    # Run in parallel
    results: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.max_proc))) as ex:
        futs = [ex.submit(run_combo, c, dry=False) for c in tasks]
        for fut in cf.as_completed(futs):
            combo_id, res = fut.result()
            results.append(res)

    # Append to CSV (resume-friendly) without pandas
    import csv

    keep_keys = {
        "combo_id",
        "scenario",
        "strategy",
        "ema_fast",
        "ema_slow",
        "rsi_len",
        "atr_len",
        "t_sep_atr",
        "pullback_atr",
        "symbol",
        "equity_end",
        "return",
        "max_dd",
        "trades",
        "error",
    }
    existing: Dict[str, Dict[str, Any]] = {}
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    cid = str(row.get("combo_id", "")).strip()
                    if cid:
                        existing[cid] = row
        except Exception:
            existing = {}

    for r in results:
        rid = str(r.get("combo_id", ""))
        reduced = {k: r.get(k) for k in keep_keys}
        existing[rid] = reduced

    fieldnames = [
        "combo_id",
        "scenario",
        "strategy",
        "symbol",
        "ema_fast",
        "ema_slow",
        "rsi_len",
        "atr_len",
        "t_sep_atr",
        "pullback_atr",
        "equity_end",
        "return",
        "max_dd",
        "trades",
        "error",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing.values():
            writer.writerow(row)

    # Print a quick top 5 summary
    try:
        def _key(row: Dict[str, Any]):
            try:
                e_val = float(row.get("equity_end") or float("nan"))
            except Exception:
                e_val = float("nan")
            try:
                t_val = int(row.get("trades") or -1)
            except Exception:
                t_val = -1
            return (e_val, t_val)

        top = sorted(existing.values(), key=_key, reverse=True)[:5]
        if top:
            print("\n=== TOP 5 (by equity_end, trades) ===")
            for r in top:
                print(
                    f"{r.get('combo_id')} | {r.get('scenario')} | equity={r.get('equity_end')} | trades={r.get('trades')} | dd={r.get('max_dd')}"
                )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
