#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from forest5.utils.io import sniff_csv_dialect, infer_ohlc_schema, normalize_ohlc_h1


def normalize_file(
    path: str | Path,
    *,
    schema: str = "auto",
    tz: str = "UTC",
    floor_to_hour: bool = False,
    weekend: str = "pad",
) -> pd.DataFrame:
    path = Path(path)
    sep, decimal, _ = sniff_csv_dialect(path)
    df = pd.read_csv(path, sep=sep, decimal=decimal)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if schema == "auto":
        df, _, _ = infer_ohlc_schema(df)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if floor_to_hour:
        df["time"] = df["time"].dt.floor("h")
    if tz:
        df["time"] = df["time"].dt.tz_convert(tz)
    df = df.drop_duplicates(subset=["time"]).sort_values("time")
    df = df.set_index("time")
    policy = "pad" if weekend == "pad" else "strict"
    if weekend == "ignore":
        df = df[df.index.dayofweek < 5]
    df = normalize_ohlc_h1(df, policy=policy)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).astype("int64")
    diffs = df.index.to_series().diff().dropna().dt.total_seconds().div(60)
    profile = {"freq_counts": diffs.value_counts().astype(int).to_dict()}
    profile_path = path.with_name(path.stem + "_profile.json")
    profile_path.write_text(json.dumps(profile))
    if any(diffs != 60):
        print("irregular intervals detected", file=sys.stderr)
    return df


def main() -> int:
    p = argparse.ArgumentParser("normalize-fxdata")
    p.add_argument("csv")
    p.add_argument("--out", default=None)
    p.add_argument("--schema", choices=["auto", "mt4", "dukas"], default="auto")
    p.add_argument("--tz", default="UTC")
    p.add_argument("--floor-to-hour", action="store_true")
    p.add_argument("--weekend", choices=["pad", "ignore"], default="pad")
    args = p.parse_args()
    df = normalize_file(
        args.csv,
        schema=args.schema,
        tz=args.tz,
        floor_to_hour=args.floor_to_hour,
        weekend=args.weekend,
    )
    if args.out:
        df.to_csv(args.out)
    else:
        df.to_csv(sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
