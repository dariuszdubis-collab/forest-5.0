"""Self-contained time-only model.

This module implements a minimal probability based trading model that relies
solely on calendar features.  For a set of prediction horizons the model
estimates the probability of future growth using historical observations and
forward chaining with an *embargo* equal to the prediction horizon.  From the
predicted probabilities quantile gates are derived which in turn are used to
generate BUY/SELL/HOLD decisions together with a position weight.

The module intentionally does not depend on any other project files so it can
be trained, serialised and loaded in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Tuple, Iterable, Any

import numpy as np
import pandas as pd
import tempfile

# ---------------------------------------------------------------------------
# Model


HORIZONS = (1, 4, 6, 24)


def _key(ts: pd.Timestamp) -> str:
    """Encode timestamp into a lookup key."""

    return f"{ts.hour}-{ts.weekday()}"


@dataclass
class TimeOnlyModel:
    """Store probability tables and quantile gates.

    Attributes
    ----------
    prob_tables:
        Mapping ``horizon -> {"hour-weekday": probability}``.
    quantiles:
        Mapping ``horizon -> (q_low, q_high)`` used for decision making.
    metadata:
        Arbitrary dictionary preserved during serialisation.
    """

    prob_tables: Dict[int, Dict[str, float]]
    quantiles: Dict[int, Tuple[float, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Prediction and decision helpers

    def predict_proba(self, ts: datetime) -> Dict[int, float]:
        """Return probability of growth for each horizon."""

        key = _key(pd.Timestamp(ts))
        return {h: self.prob_tables.get(h, {}).get(key, 0.5) for h in self.prob_tables}

    def decide(self, ts: datetime) -> Dict[str, Any]:
        """Return decision and weight for the strongest horizon.

        The returned dictionary contains:

        ``decision``
            One of ``"BUY"``, ``"SELL"`` or ``"HOLD"``.
        ``weight``
            Position weight in ``[0, 1]``.
        ``horizon``
            Horizon in hours which produced the decision.
        ``probs``
            Raw probability predictions per horizon.
        """

        probs = self.predict_proba(ts)
        candidates = {}
        for h, p in probs.items():
            q_low, q_high = self.quantiles[h]
            if p >= q_high:
                candidates[h] = ("BUY", (p - q_high) / (1 - q_high))
            elif p <= q_low:
                candidates[h] = ("SELL", (q_low - p) / q_low)
            else:
                candidates[h] = ("HOLD", 0.0)

        horizon, (decision, weight) = max(candidates.items(), key=lambda kv: abs(kv[1][1]))
        return {"decision": decision, "weight": weight, "horizon": horizon, "probs": probs}

    # ------------------------------------------------------------------
    # Serialisation helpers

    def to_json(self) -> str:
        """Serialise model to JSON string."""

        data = {
            "prob_tables": self.prob_tables,
            "quantiles": {str(h): v for h, v in self.quantiles.items()},
            "metadata": self.metadata,
        }
        return json.dumps(data)

    def save(self, path: str | Path) -> None:
        """Persist model to *path*."""

        Path(path).write_text(self.to_json())

    @classmethod
    def from_json(cls, s: str) -> "TimeOnlyModel":
        """Create model from JSON string."""

        data = json.loads(s)
        quantiles = {int(k): tuple(v) for k, v in data["quantiles"].items()}
        prob_tables = {
            int(h): {k: float(p) for k, p in tbl.items()} for h, tbl in data["prob_tables"].items()
        }
        return cls(prob_tables=prob_tables, quantiles=quantiles, metadata=data.get("metadata", {}))

    @classmethod
    def load(cls, path: str | Path) -> "TimeOnlyModel":
        """Load model from *path*."""

        return cls.from_json(Path(path).read_text())


# ---------------------------------------------------------------------------
# Training


def train(
    df: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS,
    q_low: float = 0.2,
    q_high: float = 0.8,
    return_metrics: bool = False,
) -> TimeOnlyModel | Tuple[TimeOnlyModel, Dict[int, Dict[str, float]]]:
    """Train a :class:`TimeOnlyModel`.

    Parameters
    ----------
    df:
        Input dataframe with columns ``time`` and ``y``.
    horizons:
        Iterable of integer horizons in hours.
    q_low / q_high:
        Quantile thresholds used to derive trading signals.
    return_metrics:
        If ``True`` the function returns ``(model, metrics)`` where ``metrics``
        contains accuracy and Brier score for each horizon.
    """

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df.sort_values("time", inplace=True)
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday
    df["key"] = df["hour"].astype(str) + "-" + df["weekday"].astype(str)

    prob_tables: Dict[int, Dict[str, float]] = {}
    quantiles: Dict[int, Tuple[float, float]] = {}
    metrics: Dict[int, Dict[str, float]] = {}

    for h in horizons:
        target = (df["y"].shift(-h) > df["y"]).astype(float)
        df[f"target_{h}"] = target

        grouped = df.groupby("key")
        wins = grouped[f"target_{h}"].transform(lambda s: s.fillna(0).cumsum()).shift(h)
        total = grouped.cumcount().shift(h)
        prob = wins / total
        prob[total == 0] = 0.5
        df[f"prob_{h}"] = prob

        valid = ~target.isna()
        preds = prob[valid]
        act = target[valid]

        ql = float(np.nanquantile(preds, q_low)) if len(preds.dropna()) else 0.0
        qh = float(np.nanquantile(preds, q_high)) if len(preds.dropna()) else 1.0
        quantiles[h] = (ql, qh)

        if return_metrics:
            acc = float(((preds > 0.5) == (act > 0.5)).mean())
            brier = float(np.mean((preds - act) ** 2))
            metrics[h] = {"accuracy": acc, "brier": brier}

        wins_final = grouped[f"target_{h}"].sum(min_count=1)
        total_final = grouped[f"target_{h}"].count()
        prob_tables[h] = {k: float(wins_final[k] / total_final[k]) for k in wins_final.index}

    model = TimeOnlyModel(
        prob_tables=prob_tables,
        quantiles=quantiles,
        metadata={"horizons": list(horizons), "q_low": q_low, "q_high": q_high},
    )
    return (model, metrics) if return_metrics else model


# ---------------------------------------------------------------------------
# Utilities


def self_test() -> bool:
    """Run a lightweight self-test for development purposes."""

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({"time": idx, "y": rng.normal(size=len(idx))})
    model, metrics = train(df, return_metrics=True)

    with tempfile.NamedTemporaryFile("w", delete=True) as fh:
        fh.write(model.to_json())
        fh.flush()
        _ = TimeOnlyModel.load(fh.name)

    model.decide(idx[0])
    _ = metrics
    return True


__all__ = ["TimeOnlyModel", "train", "self_test"]
