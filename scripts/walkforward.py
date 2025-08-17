#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from pandas.tseries.offsets import DateOffset

# forest5
from forest5.config import BacktestSettings, StrategySettings, RiskSettings
from forest5.backtest.engine import run_backtest


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


# ----------------------------- RANGES & GRID --------------------------------


def parse_int_range(spec: str) -> List[int]:
    """
    Parsuj zakres typu:
      - 'a-b'       -> a, a+1, ..., b
      - 'a-b:step'  -> a, a+step, ..., b
      - 'v1,v2,...' -> lista wartości
    """
    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x != ""]
    step = 1
    if ":" in spec:
        rng, st = spec.split(":")
        step = int(st)
    else:
        rng = spec
    if "-" in rng:
        a, b = rng.split("-")
        return list(range(int(a), int(b) + 1, step))
    v = int(rng)
    return [v]


def build_param_grid(
    fast: Iterable[int],
    slow: Iterable[int],
    use_rsi: bool,
    rsi_period: int,
    rsi_oversold: int,
    rsi_overbought: int,
    skip_fast_ge_slow: bool,
) -> List[Dict[str, int | bool]]:
    out: List[Dict[str, int | bool]] = []
    for f in fast:
        for s in slow:
            if skip_fast_ge_slow and f >= s:
                continue
            rec: Dict[str, int | bool] = {
                "fast": int(f),
                "slow": int(s),
                "use_rsi": bool(use_rsi),
                "rsi_period": int(rsi_period),
                "rsi_oversold": int(rsi_oversold),
                "rsi_overbought": int(rsi_overbought),
            }
            out.append(rec)
    return out


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


def iter_walkforward_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generuj kolejne okna WF: [train] -> [test], przesuwając co step_months.
    Każde okno obcinamy do maksymalnego `end`.
    """
    cur_train_start = start
    while True:
        train_end = cur_train_start + DateOffset(months=train_months) - pd.Timedelta(seconds=1)
        test_start = train_end + pd.Timedelta(seconds=1)
        test_end = test_start + DateOffset(months=test_months) - pd.Timedelta(seconds=1)

        if train_end > end:
            break
        if test_start > end:
            break

        yield (cur_train_start, train_end, test_start, min(test_end, end))

        cur_train_start = cur_train_start + DateOffset(months=step_months)
        if cur_train_start > end:
            break


# ----------------------------- EVAL & SELECTION -----------------------------


def _mk_settings(
    symbol: str,
    fast: int,
    slow: int,
    use_rsi: bool,
    rsi_period: int,
    rsi_oversold: int,
    rsi_overbought: int,
    capital: float,
    risk: float,
    fee_perc: float,
    slippage_perc: float,
    atr_period: int,
    atr_multiple: float,
) -> BacktestSettings:
    strat = StrategySettings(
        name="ema_cross",
        fast=int(fast),
        slow=int(slow),
        use_rsi=bool(use_rsi),
        rsi_period=int(rsi_period),
        rsi_oversold=int(rsi_oversold),
        rsi_overbought=int(rsi_overbought),
    )
    riskset = RiskSettings(
        initial_capital=float(capital),
        risk_per_trade=float(risk),
        fee_perc=float(fee_perc),
        slippage_perc=float(slippage_perc),
    )
    return BacktestSettings(
        symbol=symbol,
        strategy=strat,
        risk=riskset,
        atr_period=int(atr_period),
        atr_multiple=float(atr_multiple),
    )


def evaluate_df(
    df: pd.DataFrame,
    settings: BacktestSettings,
) -> Tuple[float, float, int, float]:
    res = run_backtest(df, settings)
    eq_end = float(res.equity_curve.iloc[-1]) if len(res.equity_curve) else 0.0
    init_cap = float(settings.risk.initial_capital)
    ret = (eq_end / init_cap) - 1.0 if init_cap > 0 else 0.0
    max_dd = float(res.max_dd)
    trades = len(getattr(res.trades, "trades", getattr(res.trades, "__iter__", [])))
    return ret, max_dd, int(trades), eq_end


def pick_best_on_train(
    train_df: pd.DataFrame,
    grid: List[Dict[str, int | bool]],
    symbol: str,
    capital: float,
    risk: float,
    fee_perc: float,
    slippage_perc: float,
    atr_period: int,
    atr_multiple: float,
    dd_penalty: float,
) -> Tuple[Dict[str, int | bool], float, float]:
    best: Dict[str, int | bool] | None = None
    best_score = -1e9
    best_ret = 0.0
    best_dd = 1.0

    for p in grid:
        st = _mk_settings(
            symbol=symbol,
            fast=int(p["fast"]),
            slow=int(p["slow"]),
            use_rsi=bool(p["use_rsi"]),
            rsi_period=int(p["rsi_period"]),
            rsi_oversold=int(p["rsi_oversold"]),
            rsi_overbought=int(p["rsi_overbought"]),
            capital=capital,
            risk=risk,
            fee_perc=fee_perc,
            slippage_perc=slippage_perc,
            atr_period=atr_period,
            atr_multiple=atr_multiple,
        )
        tr_ret, tr_dd, _, _ = evaluate_df(train_df, st)
        score = tr_ret - dd_penalty * tr_dd
        if (score > best_score) or (score == best_score and (tr_ret > best_ret or tr_dd < best_dd)):
            best = p
            best_score = score
            best_ret = tr_ret
            best_dd = tr_dd

    assert best is not None  # grid nie może być puste
    return best, best_ret, best_dd


# ----------------------------- CLI / MAIN -----------------------------------


def main() -> None:
    ap = argparse.ArgumentParser("walkforward")
    ap.add_argument("--csv", required=True, help="Ścieżka do danych (CSV)")
    ap.add_argument("--symbol", default="SYMBOL")

    ap.add_argument("--fast", default="5-20")
    ap.add_argument("--slow", default="20-60:5")

    ap.add_argument("--use-rsi", action="store_true")
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--rsi-oversold", type=int, default=30)
    ap.add_argument("--rsi-overbought", type=int, default=70)

    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--fee-perc", type=float, default=0.0005)
    ap.add_argument("--slippage-perc", type=float, default=0.0)

    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--atr-multiple", type=float, default=2.0)

    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)

    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--step-months", type=int, default=3)

    ap.add_argument("--dd-penalty", type=float, default=0.5)
    ap.add_argument("--skip-fast-ge-slow", action="store_true")
    ap.add_argument("--export", default="out/walkforward.csv")

    args = ap.parse_args()

    # 1) dane
    df = load_prices(args.csv)
    if args.start or args.end:
        df = cut_df(df, args.start, args.end)
    if df.empty:
        raise SystemExit("Pusty zakres danych po cięciu start/end.")

    # 2) grid parametrów
    fast_vals = parse_int_range(args.fast)
    slow_vals = parse_int_range(args.slow)
    grid = build_param_grid(
        fast_vals,
        slow_vals,
        args.use_rsi,
        args.rsi_period,
        args.rsi_oversold,
        args.rsi_overbought,
        args.skip_fast_ge_slow,
    )
    if not grid:
        raise SystemExit("Siatka parametrów jest pusta (sprawdź --fast/--slow).")

    # 3) walk-forward
    start_ts = df.index[0]
    end_ts = df.index[-1]
    if args.start:
        start_ts = max(start_ts, ts_naive_utc(args.start))
    if args.end:
        end_ts = min(end_ts, ts_naive_utc(args.end))

    rows: List[Dict[str, object]] = []

    for tr_start, tr_end, te_start, te_end in iter_walkforward_windows(
        start_ts, end_ts, args.train_months, args.test_months, args.step_months
    ):
        train = df.loc[(df.index >= tr_start) & (df.index <= tr_end)]
        test = df.loc[(df.index >= te_start) & (df.index <= te_end)]
        if train.empty or test.empty:
            continue

        best, tr_ret, tr_dd = pick_best_on_train(
            train_df=train,
            grid=grid,
            symbol=args.symbol,
            capital=args.capital,
            risk=args.risk,
            fee_perc=args.fee_perc,
            slippage_perc=args.slippage_perc,
            atr_period=args.atr_period,
            atr_multiple=args.atr_multiple,
            dd_penalty=args.dd_penalty,
        )

        # ewaluacja na teście
        st_best = _mk_settings(
            symbol=args.symbol,
            fast=int(best["fast"]),
            slow=int(best["slow"]),
            use_rsi=bool(best["use_rsi"]),
            rsi_period=int(best["rsi_period"]),
            rsi_oversold=int(best["rsi_oversold"]),
            rsi_overbought=int(best["rsi_overbought"]),
            capital=args.capital,
            risk=args.risk,
            fee_perc=args.fee_perc,
            slippage_perc=args.slippage_perc,
            atr_period=args.atr_period,
            atr_multiple=args.atr_multiple,
        )
        te_ret, te_dd, te_trades, te_eq_end = evaluate_df(test, st_best)

        row = {
            "train_start": tr_start,
            "train_end": tr_end,
            "test_start": te_start,
            "test_end": te_end,
            "fast": int(best["fast"]),
            "slow": int(best["slow"]),
            "use_rsi": bool(best["use_rsi"]),
            "rsi_period": int(best["rsi_period"]),
            "rsi_oversold": int(best["rsi_oversold"]),
            "rsi_overbought": int(best["rsi_overbought"]),
            "train_ret": tr_ret,
            "train_max_dd": tr_dd,
            "test_ret": te_ret,
            "test_max_dd": te_dd,
            "test_trades": te_trades,
            "test_equity_end": te_eq_end,
        }
        rows.append(row)
        print(
            {
                "event": "wf_window",
                "train": [str(tr_start), str(tr_end)],
                "test": [str(te_start), str(te_end)],
                "best": {"fast": row["fast"], "slow": row["slow"], "use_rsi": row["use_rsi"]},
                "train_ret": tr_ret,
                "train_dd": tr_dd,
                "test_ret": te_ret,
                "test_dd": te_dd,
            }
        )

    # 4) eksport
    out_path = Path(args.export)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print({"event": "walkforward_export_done", "out": str(out_path), "rows": len(rows)})


if __name__ == "__main__":
    main()
