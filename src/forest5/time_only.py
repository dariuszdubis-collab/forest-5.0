from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import argparse
import json
from pathlib import Path
import tempfile
from typing import Dict, Tuple, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


@dataclass
class TimeOnlyModel:
    """Simple time-of-day model based on quantile gates.

    Historically the model used ``quantile_gates`` along with ``q_low`` and
    ``q_high`` fields.  The new API exposes the same information via
    ``prob_tables`` and ``quantiles`` while keeping backwards compatibility with
    the previous names.  ``prob_tables`` maps an hour of day to a pair of
    quantile thresholds and ``quantiles`` stores the low/high quantile values
    used during training.
    """

    prob_tables: Dict[int, Tuple[float, float]]
    quantiles: Tuple[float, float]
    win_rates: Dict[int, float] | None = None

    def __init__(
        self,
        prob_tables: Dict[int, Tuple[float, float]] | None = None,
        quantiles: Tuple[float, float] | None = None,
        *,
        quantile_gates: Dict[int, Tuple[float, float]] | None = None,
        q_low: float | None = None,
        q_high: float | None = None,
        win_rates: Dict[int, float] | None = None,
    ) -> None:
        if prob_tables is None and quantile_gates is not None:
            prob_tables = quantile_gates
        if quantiles is None and q_low is not None and q_high is not None:
            quantiles = (q_low, q_high)
        if prob_tables is None or quantiles is None:
            raise TypeError("prob_tables and quantiles must be provided")
        self.prob_tables = {int(k): tuple(v) for k, v in prob_tables.items()}
        self.quantiles = (float(quantiles[0]), float(quantiles[1]))
        self.win_rates = win_rates or {}

    # ------------------------------------------------------------------
    # Backwards-compatible attribute aliases
    # ------------------------------------------------------------------
    @property
    def quantile_gates(self) -> Dict[int, Tuple[float, float]]:
        return self.prob_tables

    @property
    def q_low(self) -> float:
        return self.quantiles[0]

    @property
    def q_high(self) -> float:
        return self.quantiles[1]

    def decide(
        self, ts: datetime, value: float | None = None
    ) -> dict[str, float | str] | Literal["BUY", "SELL", "WAIT"]:
        """Return BUY/SELL/WAIT decision based on quantile gates.

        When ``value`` is provided the legacy string-based API is used.
        For the new timestamp-only API a mapping with ``decision`` and
        ``confidence`` keys is returned.
        """
        hour = ts.hour
        gates = self.quantile_gates.get(hour)
        if gates is None or value is None:
            res = "WAIT"
        else:
            low, high = gates
            if value <= low:
                res = "SELL"
            elif value >= high:
                res = "BUY"
            else:
                res = "WAIT"
        if value is None:
            conf = 1.0
            if self.win_rates is not None:
                conf = self.win_rates.get(hour, 0.0)
            return {"decision": res, "confidence": conf}
        return res

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    def to_json(self) -> str:
        data = {
            "prob_tables": {str(k): v for k, v in self.prob_tables.items()},
            "quantiles": list(self.quantiles),
            "win_rates": self.win_rates or {},
        }
        return json.dumps(data)

    @classmethod
    def load(cls, path: str | Path) -> "TimeOnlyModel":
        data = json.loads(Path(path).read_text())
        if "prob_tables" in data:
            tables = {int(k): tuple(v) for k, v in data["prob_tables"].items()}
            quantiles = tuple(data.get("quantiles", [0.0, 1.0]))  # type: ignore[assignment]
        else:  # legacy format
            tables = {int(k): tuple(v) for k, v in data["quantile_gates"].items()}
            quantiles = (data["q_low"], data["q_high"])
        win_rates = {int(k): float(v) for k, v in data.get("win_rates", {}).items()}
        return cls(prob_tables=tables, quantiles=quantiles, win_rates=win_rates)


def _cumulative_counts(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Calculate cumulative win/total counts shifted by ``h + 1`` hours.

    The dataframe must already contain the ``hour`` column.
    ``y`` values greater than zero are considered wins.
    """

    df = df.copy()
    df["win"] = (df["y"] > 0).astype(int)
    df["cum_wins"] = df.groupby("hour")["win"].cumsum()
    df["cum_total"] = df.groupby("hour").cumcount() + 1
    df["wins"] = df["cum_wins"].shift(h + 1)
    df["total"] = df["cum_total"].shift(h + 1)
    return df


def train(
    df: pd.DataFrame,
    q_low: float = 0.1,
    q_high: float = 0.9,
    h: int = 0,
) -> TimeOnlyModel:
    """Train quantile gates on the provided dataframe.

    The dataframe must contain `time` (datetime-like) and `y` columns where `y`
    is the target variable used for the decision.  When ``h`` is positive the
    returned model also includes historical win rates per hour where the counts
    are shifted by ``h + 1`` bars ensuring predictions at time ``t`` use only
    data up to ``t - h - 1``.
    """

    df = df.copy()
    if not is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].dt.tz_localize(None)
    df["hour"] = df["time"].dt.hour

    stats = _cumulative_counts(df, h)
    win_rates: Dict[int, float] = {}
    for hour, sub in stats.groupby("hour"):
        last = sub[["wins", "total"]].iloc[-1]
        wins = last["wins"]
        total = last["total"]
        if pd.notna(wins) and pd.notna(total) and total > 0:
            win_rates[int(hour)] = float(wins / total)
        else:
            win_rates[int(hour)] = 0.0

    gates: Dict[int, Tuple[float, float]] = {}
    for hour, series in df.groupby("hour")["y"]:
        gates[int(hour)] = (float(series.quantile(q_low)), float(series.quantile(q_high)))
    return TimeOnlyModel(
        prob_tables=gates,
        quantiles=(q_low, q_high),
        win_rates=win_rates,
    )


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
        raise ValueError("Quantile gates mismatch after serialization")
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
