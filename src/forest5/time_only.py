from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import argparse
import json
from pathlib import Path
import tempfile
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class TimeOnlyModel:
    """Simple time-of-day quantile gates.

    For each hour of day we store low/high quantile thresholds of the target
    variable. The decision uses only the timestamp and the observed value.
    """

    quantile_gates: Dict[int, Tuple[float, float]]
    q_low: float
    q_high: float

    def decide(self, ts: datetime, value: float) -> str:
        """Return BUY/SELL/WAIT decision based on quantile gates."""
        hour = ts.hour
        gates = self.quantile_gates.get(hour)
        if gates is None:
            return "WAIT"
        low, high = gates
        if value <= low:
            return "SELL"
        if value >= high:
            return "BUY"
        return "WAIT"

    def save(self, path: str | Path) -> None:
        data = {
            "quantile_gates": {str(k): v for k, v in self.quantile_gates.items()},
            "q_low": self.q_low,
            "q_high": self.q_high,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "TimeOnlyModel":
        data = json.loads(Path(path).read_text())
        gates = {int(k): tuple(v) for k, v in data["quantile_gates"].items()}
        return cls(quantile_gates=gates, q_low=data["q_low"], q_high=data["q_high"])


def train(df: pd.DataFrame, q_low: float = 0.1, q_high: float = 0.9) -> TimeOnlyModel:
    """Train quantile gates on the provided dataframe.

    The dataframe must contain `time` (datetime-like) and `y` columns where `y`
    is the target variable used for the decision.
    """

    df = df.copy()
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    gates: Dict[int, Tuple[float, float]] = {}
    for hour, series in df.groupby("hour")["y"]:
        gates[int(hour)] = (float(series.quantile(q_low)), float(series.quantile(q_high)))
    return TimeOnlyModel(quantile_gates=gates, q_low=q_low, q_high=q_high)


def self_test() -> bool:
    """Run a lightweight self-test.

    The test trains a model on random data, saves artifacts to a temporary
    location and loads them back verifying the round-trip.
    """

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    df = pd.DataFrame({"time": idx, "y": rng.normal(size=len(idx))})
    model = train(df)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "time_only.json"
        model.save(path)
        loaded = TimeOnlyModel.load(path)
    if model.quantile_gates != loaded.quantile_gates:
        raise RuntimeError("Quantile gates mismatch after serialization")
    # exercise decision logic on a single sample
    loaded.decide(idx[0], df["y"].iloc[0])
    return True


def _run_cli() -> int:
    """Console entry-point for training or running the self-test."""

    p = argparse.ArgumentParser("time-only")
    p.add_argument("csv", nargs="?", help="CSV with time,y columns")
    p.add_argument("--out", help="Where to save artifacts")
    p.add_argument("--q-low", type=float, default=0.1)
    p.add_argument("--q-high", type=float, default=0.9)
    p.add_argument("--self-test", action="store_true", help="run module self-test")
    args = p.parse_args()

    if args.self_test:
        ok = self_test()
        print("self-test: OK" if ok else "self-test: FAILED")
        return 0 if ok else 1

    if not args.csv or not args.out:
        p.error("csv and --out are required unless --self-test is used")

    df = pd.read_csv(args.csv, parse_dates=["time"])
    model = train(df, args.q_low, args.q_high)
    model.save(args.out)
    print(f"Saved artifacts -> {args.out}")
    return 0


__all__ = ["TimeOnlyModel", "train", "self_test", "_run_cli"]
