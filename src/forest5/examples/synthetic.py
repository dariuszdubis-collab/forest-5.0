from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def generate_ohlc(periods: int = 100, start_price: float = 100.0, freq: str = "D") -> pd.DataFrame:
    # normalize frequency to lowercase to avoid pandas warnings
    freq = freq.lower()
    idx = pd.date_range("2024-01-01", periods=periods, freq=freq)
    rnd = np.random.default_rng(42)
    ret = rnd.normal(0, 0.01, size=periods)
    px = start_price * (1 + pd.Series(ret, index=idx)).cumprod()
    high = px * (1 + rnd.uniform(0.0, 0.02, size=periods))
    low = px * (1 - rnd.uniform(0.0, 0.02, size=periods))
    open_ = px.shift(1).fillna(px.iloc[0])
    close = px
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    df.index.name = "time"
    return df


def main() -> None:
    p = argparse.ArgumentParser("forest5-demo")
    p.add_argument("--periods", type=int, default=50)
    p.add_argument("--start", type=float, default=100.0)
    p.add_argument("--freq", default="D")
    p.add_argument("--out")
    args = p.parse_args()
    df = generate_ohlc(args.periods, args.start, args.freq)
    if args.out:
        df.to_csv(args.out, index=True, date_format="%Y-%m-%d %H:%M:%S")
        print(f"Saved -> {args.out}")
    else:
        print(df.head(10))


if __name__ == "__main__":
    main()
