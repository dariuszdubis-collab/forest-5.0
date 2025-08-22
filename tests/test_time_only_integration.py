from __future__ import annotations

from pathlib import Path

import pandas as pd

from forest5.time_only import TimeOnlyModel, train


def _synthetic_df() -> pd.DataFrame:
    """Create a simple dataset with deterministic quantiles."""
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    # values 0..3 -> q0.25=0.75, q0.75=2.25 for hour 0
    return pd.DataFrame({"time": idx, "y": [0.0, 1.0, 2.0, 3.0]})


def test_train_decide_and_serialize(tmp_path: Path) -> None:
    df = _synthetic_df()
    model = train(df)

    ts = df["time"].iloc[0]
    decision = model.decide(ts)["decision"]
    assert decision in {"BUY", "SELL", "WAIT"}

    artifact = tmp_path / "time_only.json"
    model.save(artifact)
    loaded = TimeOnlyModel.load(artifact)
    loaded.decide(ts)["decision"]
