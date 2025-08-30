#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from typing import Any

import pandas as pd

from forest5.utils.io import read_ohlc_csv
from forest5.signals.h1_ema_rsi_atr import compute_primary_signal_h1
from forest5.utils.debugger import TraceCollector


def run_debug(
    csv: str,
    start: str | None,
    ema_fast: int,
    ema_slow: int,
    rsi_len: int,
    atr_len: int,
    t_sep_atr: float,
    pullback_atr: float,
    entry_buffer_atr: float,
    sl_min_atr: float,
    rr: float,
) -> dict[str, Any]:
    df = read_ohlc_csv(csv)
    if start:
        df = df.loc[pd.Timestamp(start) :]

    params = dict(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_len,
        atr_period=atr_len,
        t_sep_atr=t_sep_atr,
        pullback_atr=pullback_atr,
        entry_buffer_atr=entry_buffer_atr,
        sl_min_atr=sl_min_atr,
        rr=rr,
    )

    events: list[dict] = []
    counts = Counter()
    armed = 0
    triggers = 0
    for i in range(len(df)):
        collector = TraceCollector()
        _ = compute_primary_signal_h1(df.iloc[: i + 1], params, collector=collector)
        for ev in collector.events:
            # ev: {time,stage,reason,...}
            events.append(ev)
            stage = ev.get("stage")
            reason = ev.get("reason")
            if stage == "setup_candidate" and reason:
                counts[reason] += 1
            if stage == "setup_trigger" and reason == "armed":
                armed += 1
            if stage == "pattern" and reason == "pattern_trigger_hit":
                counts["pattern_ok"] += 1
            if stage == "pattern" and reason == "pattern_trigger_miss":
                counts["pattern_miss"] += 1

    # Aggregate reasons
    summary = {
        "total_bars": len(df),
        "armed": armed,
        "counts": dict(counts),
    }
    # Top 10 most recent candidate notes for inspection
    sample = [e for e in events if e.get("stage") in ("setup_candidate", "pattern")][-10:]
    summary["sample_tail"] = sample
    return summary


def main() -> int:
    ap = argparse.ArgumentParser("debug-h1-decision")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--ema-fast", type=int, default=34)
    ap.add_argument("--ema-slow", type=int, default=89)
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--atr-len", type=int, default=14)
    ap.add_argument("--t-sep-atr", type=float, default=0.5)
    ap.add_argument("--pullback-atr", type=float, default=0.5)
    ap.add_argument("--entry-buffer-atr", type=float, default=0.1)
    ap.add_argument("--sl-min-atr", type=float, default=0.0)
    ap.add_argument("--rr", type=float, default=2.0)
    args = ap.parse_args()

    out = run_debug(
        csv=args.csv,
        start=args.start,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        rsi_len=args.rsi_len,
        atr_len=args.atr_len,
        t_sep_atr=args.t_sep_atr,
        pullback_atr=args.pullback_atr,
        entry_buffer_atr=args.entry_buffer_atr,
        sl_min_atr=args.sl_min_atr,
        rr=args.rr,
    )
    # Pretty print
    print("Bars:", out["total_bars"], "Armed:", out["armed"]) 
    print("Reasons:")
    for k, v in sorted(out["counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")
    print("\nSample tail events:")
    for e in out["sample_tail"]:
        print(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
