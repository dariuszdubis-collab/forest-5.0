from forest5.utils.timeframes import normalize_timeframe
import pytest


def test_normalize_timeframe_variants():
    assert normalize_timeframe("H") == "1h"
    assert normalize_timeframe("1H") == "1h"
    assert normalize_timeframe("60min") == "1h"
    assert normalize_timeframe("240") == "4h"


def test_rare_timeframe_aliases():
    assert normalize_timeframe("M") == "1m"
    assert normalize_timeframe("1D") == "1d"
    assert normalize_timeframe("1440") == "1d"


@pytest.mark.parametrize("bad", ["weird", "", "13", "1X"])
def test_invalid_timeframe(bad):
    with pytest.raises(ValueError):
        normalize_timeframe(bad)
