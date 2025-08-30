#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from forest5.config import BacktestSettings
from forest5.backtest.engine import run_backtest
from forest5.utils.io import read_ohlc_csv
from forest5.utils.timeindex import ensure_h1
from forest5.utils.io import normalize_ohlc_h1


def run_single(
    csv: Path,
    symbol: str,
    start: str,
    *,
    ema_fast: int,
    ema_slow: int,
    rsi_len: int,
    atr_len: int,
    t_sep_atr: float,
    pullback_atr: float,
    entry_buffer_atr: float,
    rr: float,
    entry_mode: str,
    ttl_bars: int,
    use_ema_gates: bool,
) -> dict:
    df = read_ohlc_csv(csv)
    df = df.loc[pd.Timestamp(start) :]
    df, _ = ensure_h1(df, policy="pad")
    df = normalize_ohlc_h1(df, policy="pad")

    settings = BacktestSettings(
        symbol=symbol,
        timeframe="1h",
        strategy=dict(
            name="h1_ema_rsi_atr",
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            atr_period=atr_len,
            rsi_period=rsi_len,
            t_sep_atr=t_sep_atr,
            pullback_atr=pullback_atr,
            entry_buffer_atr=entry_buffer_atr,
            rr=rr,
            entry_mode=entry_mode,
            use_ema_gates=use_ema_gates,
        ),
        setup_ttl_bars=int(ttl_bars),
    )
    res = run_backtest(df, settings)
    eq_end = float(res.equity_curve.iloc[-1]) if not res.equity_curve.empty else 0.0
    trades = len(res.trades.trades)
    ret = eq_end / settings.risk.initial_capital - 1.0
    return dict(
        symbol=symbol,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        t_sep_atr=t_sep_atr,
        pullback_atr=pullback_atr,
        entry_buffer_atr=entry_buffer_atr,
        rr=rr,
        entry_mode=entry_mode,
        ttl_bars=ttl_bars,
        use_ema_gates=use_ema_gates,
        equity_end=eq_end,
        return_pct=ret,
        max_dd=float(res.max_dd),
        trades=trades,
    )


def main() -> int:
    ap = argparse.ArgumentParser("h1-mini-grid")
    ap.add_argument("--data-dir", type=Path, default=Path("repos/forest-5.0/Fxdata"))
    ap.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY", "EURJPY"])
    ap.add_argument("--from", dest="start", default="2024-01-01")
    ap.add_argument("--out", type=Path, default=Path("repos/forest-5.0/out_runs/h1_mini_grid.csv"))
    args = ap.parse_args()

    combos = []
    for use_ema in (True, False):
        for entry_mode in ("breakout", "close", "close_next"):
            for entry_buffer_atr in (0.0, 0.05):
                combos.append(
                    dict(use_ema_gates=use_ema, entry_mode=entry_mode, entry_buffer_atr=entry_buffer_atr)
                )

    rows: list[dict] = []
    for sym in args.symbols:
        csv = args.data_dir / f"{sym}_H1.csv"
        for cmb in combos:
            row = run_single(
                csv,
                sym,
                args.start,
                ema_fast=12,
                ema_slow=34,
                rsi_len=14,
                atr_len=14,
                t_sep_atr=0.2,
                pullback_atr=1.2,
                entry_buffer_atr=cmb["entry_buffer_atr"],
                rr=1.5,
                entry_mode=cmb["entry_mode"],
                ttl_bars=2,
                use_ema_gates=cmb["use_ema_gates"],
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
