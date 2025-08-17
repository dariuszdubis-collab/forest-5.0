from forest5.utils.timeframes import normalize_timeframe
import pytest


def test_normalize_timeframe_variants():
    assert normalize_timeframe("H") == "1h"
    assert normalize_timeframe("1H") == "1h"
    assert normalize_timeframe("240") == "4h"


def test_rare_timeframe_aliases():
    assert normalize_timeframe("M") == "1m"
    assert normalize_timeframe("1D") == "1d"
    assert normalize_timeframe("1440") == "1d"


def test_normalize_timeframe_60min_alias():
    assert normalize_timeframe("60min") == "1h"


@pytest.mark.parametrize("tf, expected", [("240", "4h"), ("2H", "2h")])
def test_normalize_timeframe_numeric_and_hour_aliases(tf, expected):
    assert normalize_timeframe(tf) == expected


@pytest.mark.parametrize("bad", ["weird", "", "13", "1X", "999x"])
def test_invalid_timeframe(bad):
    with pytest.raises(ValueError):
        normalize_timeframe(bad)
