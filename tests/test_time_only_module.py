from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forest5.time_only import TimeOnlyModel, train, self_test


def test_self_test_runs() -> None:
    assert self_test()  # nosec B101


def test_train_decide_and_serialize(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    df = pd.DataFrame({"time": idx, "y": rng.normal(size=len(idx))})

    model, metrics = train(df, return_metrics=True)
    assert metrics and set(model.prob_tables)  # nosec B101

    ts = idx[-1]
    decision = model.decide(ts)
    assert set(decision) == {"decision", "weight", "horizon", "probs"}  # nosec B101

    artifact = tmp_path / "time_only.json"
    model.save(artifact)
    loaded = TimeOnlyModel.load(artifact)
    assert loaded.decide(ts)["decision"] in {"BUY", "SELL", "HOLD"}  # nosec B101
