from forest5.utils.timeframes import normalize_timeframe
import pytest


def test_normalize_timeframe_variants():
    assert normalize_timeframe("H") == "1h"
    assert normalize_timeframe("1H") == "1h"
    assert normalize_timeframe("60min") == "1h"
    assert normalize_timeframe("240") == "4h"


def test_invalid_timeframe():
    with pytest.raises(ValueError):
        normalize_timeframe("weird")
