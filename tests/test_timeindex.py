import pandas as pd
import pytest

from forest5.utils.timeindex import ensure_h1


def _sample_df():
    idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 03:00"])
    return pd.DataFrame(
        {
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
        },
        index=idx,
    )


def test_ensure_h1_strict_raises():
    with pytest.raises(ValueError):
        ensure_h1(_sample_df(), policy="strict")


def test_ensure_h1_pad_reports_gap():
    out, meta = ensure_h1(_sample_df(), policy="pad")
    assert len(out) == 4
    assert len(meta["gaps"]) == 1
    assert meta["median_bar_minutes"] == 180.0
    gap = meta["gaps"][0]
    assert gap.missing == 2
    assert gap.start == pd.Timestamp("2020-01-01 00:00", tz="UTC")
    assert gap.end == pd.Timestamp("2020-01-01 03:00", tz="UTC")
