from __future__ import annotations

import pandas as pd
import pytest

from forest5 import time_only


def test_training_window_and_prediction_horizon_do_not_overlap() -> None:
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=6, freq="D"),
            "y": [1, 1, 1, 1, 0, 0],
        }
    )
    h = 1
    model = time_only.train(df, h=h)

    # last two samples should not contribute to training stats
    assert model.win_rates[0] == pytest.approx(1.0)

    df2 = df.copy()
    df2["hour"] = df2["time"].dt.hour
    stats = time_only._cumulative_counts(df2, h)

    # first h + 1 rows have no training data available
    assert stats["wins"].isna().tolist()[: h + 1] == [True, True]
    assert stats["wins"].iloc[h + 1 :].tolist() == [1, 2, 3, 4]
    assert stats["total"].iloc[h + 1 :].tolist() == [1, 2, 3, 4]
