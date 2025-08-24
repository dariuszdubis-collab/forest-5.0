#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

# forest5
from forest5.cli import SafeHelpFormatter, _parse_range
from forest5.utils.argparse_ext import PercentAction
from forest5.backtest.walkforward import run_walkforward


# ----------------------------- CSV LOADING ---------------------------------


_TIME_CANDIDATES = ("time", "timestamp", "date", "datetime")


def _detect_time_col(path: str | Path) -> str:
    """Znajdź kolumnę czasu (case-insensitive)."""
    hdr = pd.read_csv(path, nrows=0)
    cols = list(hdr.columns)
    lower = [c.strip().lower() for c in cols]
    for cand in _TIME_CANDIDATES:
        if cand in lower:
            return cols[lower.index(cand)]
    # fallback – poszukaj wzorca w nazwie
    for i, c in enumerate(lower):
        if "time" in c or "date" in c:
            return cols[i]
    raise ValueError(f"Nie znaleziono kolumny czasu w {path}. Kolumny: {list(hdr.columns)}")


def _normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ujednolić nazwy OHLC do lower-case: open, high, low, close."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if {"open", "high", "low", "close"}.issubset(df.columns):
        return df

    # mapowania alternatywnych nazw spotykanych w FX CSV
    alt_map: Dict[str, str] = {}
    for c in list(df.columns):
        lc = c.lower()
        if lc in {"bidopen", "openbid", "o", "askopen"}:
            alt_map[c] = "open"
        elif lc in {"bidhigh", "h", "askhigh"}:
            alt_map[c] = "high"
        elif lc in {"bidlow", "l", "asklow"}:
            alt_map[c] = "low"
        elif lc in {"bidclose", "closebid", "c", "askclose"}:
            alt_map[c] = "close"
    if alt_map:
        df = df.rename(columns=alt_map)

    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"Brak kolumn OHLC {missing}; dostępne: {list(df.columns)}")
    return df


def load_prices(path: str | Path) -> pd.DataFrame:
    """
    Wczytaj CSV:
    - wykryj kolumnę czasu,
    - sparsuj do UTC i zdejmij strefę (tz-naive index),
    - ujednolić nazwy OHLC,
    - posortuj po czasie i usuń duplikaty.
    """
    tcol = _detect_time_col(path)
    raw = pd.read_csv(path)

    dt = pd.to_datetime(raw[tcol], utc=True, errors="raise").dt.tz_convert(None)
    df = raw.drop(columns=[tcol]).copy()
    df.index = pd.Index(dt, name="time")

    df = _normalize_ohlc_columns(df)
    df = df[["open", "high", "low", "close"]].astype(float)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    return df


# ----------------------------- TIME UTILS -----------------------------------


def ts_naive_utc(s: str) -> pd.Timestamp:
    """Parse do UTC i zdejmij strefę (naive)."""
    return pd.to_datetime(s, utc=True).tz_convert(None)


def cut_df(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df
    if start:
        s = ts_naive_utc(start)
        out = out.loc[out.index >= s]
    if end:
        e = ts_naive_utc(end)
        out = out.loc[out.index <= e]
    return out


# ----------------------------- CLI / MAIN -----------------------------------


def main() -> None:
    ap = argparse.ArgumentParser("walkforward", formatter_class=SafeHelpFormatter)
    ap.add_argument("--csv", required=True, help="Ścieżka do danych (CSV)")
    ap.add_argument("--symbol", default="SYMBOL")

    ap.add_argument("--fast", default="5-20")
    ap.add_argument("--slow", default="20-60:5")

    ap.add_argument("--use-rsi", action="store_true")
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--rsi-oversold", type=int, default=30, choices=range(0, 101))
    ap.add_argument("--rsi-overbought", type=int, default=70, choices=range(0, 101))

    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--risk", action=PercentAction, default=0.01)
    ap.add_argument("--fee-perc", action=PercentAction, default=0.0005)
    ap.add_argument("--slippage-perc", action=PercentAction, default=0.0)

    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--atr-multiple", type=float, default=2.0)

    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)

    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--step-months", type=int, default=3)
    ap.add_argument("--mode", choices=("rolling", "anchored"), default="rolling")

    ap.add_argument("--skip-fast-ge-slow", action="store_true")
    ap.add_argument("--out", default="out/walkforward")
    ap.add_argument("--top", type=int, default=10, help="Ile rekordów wyświetlić")

    args = ap.parse_args()

    # 1) dane
    df = load_prices(args.csv)
    if args.start or args.end:
        df = cut_df(df, args.start, args.end)
    if df.empty:
        raise SystemExit("Pusty zakres danych po cięciu start/end.")

    # 2) param ranges
    fast_vals = _parse_range(args.fast)
    slow_vals = _parse_range(args.slow)

    res_df = run_walkforward(
        df,
        symbol=args.symbol,
        fast_values=fast_vals,
        slow_values=slow_vals,
        use_rsi=args.use_rsi,
        rsi_period=args.rsi_period,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        capital=args.capital,
        risk=args.risk,
        fee_perc=args.fee_perc,
        slippage_perc=args.slippage_perc,
        atr_period=args.atr_period,
        atr_multiple=args.atr_multiple,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        mode=args.mode,
        skip_fast_ge_slow=args.skip_fast_ge_slow,
        out_dir=args.out,
    )

    sort_cols = [c for c in ("test_pnl_net", "test_sharpe") if c in res_df.columns]
    if sort_cols:
        res_df = res_df.sort_values(by=sort_cols, ascending=False)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(res_df.head(args.top).to_string(index=False, justify="right"))

    print(
        {
            "event": "walkforward_export_done",
            "out": str(Path(args.out) / "summary.csv"),
            "rows": len(res_df),
        }
    )


if __name__ == "__main__":
    main()
