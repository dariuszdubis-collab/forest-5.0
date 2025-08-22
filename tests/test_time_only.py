from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from forest5 import time_only


def test_self_test_runs() -> None:
    assert time_only.self_test()


def test_artifact_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    df = pd.DataFrame({"time": idx, "y": rng.normal(size=len(idx))})
    model = time_only.train(df)
    artifact_path = tmp_path / "time_only.json"
    model.save(artifact_path)
    loaded = time_only.TimeOnlyModel.load(artifact_path)
    data = artifact_path.read_text()
    assert "prob_tables" in data and "quantiles" in data
    # exercise a decision to ensure load works
    loaded.decide(idx[0])["decision"]


def test_train_handles_timezone() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"time": idx, "y": rng.normal(size=len(idx))})
    model = time_only.train(df)
    assert set(model.prob_tables.keys()) == set(range(24))
