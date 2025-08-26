import pandas as pd
import pytest

from forest5.utils.timeindex import ensure_h1


def _gap_df():
    idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 02:00"])
    return pd.DataFrame({"open": [1.0, 1.2]}, index=idx)


def test_strict_raises_on_gap():
    with pytest.raises(ValueError):
        ensure_h1(_gap_df(), policy="strict")


def test_pad_inserts_missing_row():
    out, meta = ensure_h1(_gap_df(), policy="pad")
    assert len(out) == 3
    assert meta["gaps"][0].missing == 1
    assert meta["median_bar_minutes"] == 120.0
    assert out.isna().any().any()


def test_drop_discards_missing_period():
    out, meta = ensure_h1(_gap_df(), policy="drop")
    assert len(out) == 2
    assert meta["gaps"][0].missing == 1
    assert meta["median_bar_minutes"] == 120.0
