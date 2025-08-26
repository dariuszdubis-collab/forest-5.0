import pandas as pd
from structlog.testing import capture_logs

from forest5.utils.io import read_ohlc_csv_smart
from forest5.utils.timeindex import ensure_h1
from forest5.utils.log import setup_logger, E_DATA_CSV_SCHEMA, E_DATA_TIME_GAPS


def test_read_ohlc_csv_smart_emits_event(tmp_path):
    setup_logger()
    path = tmp_path / "aliases.csv"
    path.write_text(
        "Date,Time,O,H,L,C,V\n" "2020-01-01,00:00,1,2,0,1,5\n",
        encoding="utf-8",
    )
    with capture_logs() as cap:
        read_ohlc_csv_smart(path)
    events = [e for e in cap if e.get("event") == E_DATA_CSV_SCHEMA]
    assert events
    ev = events[0]
    assert ev["separator"] == ","
    assert ev["decimal"] == "."
    assert ev["has_header"] is True
    assert ev["path"] == str(path)
    assert ev["rows"] == 1
    assert ev["aliases"]["open"] == "O"
    assert ev["from"] == "2020-01-01T00:00:00+00:00"
    assert ev["to"] == "2020-01-01T00:00:00+00:00"


def test_ensure_h1_emits_event():
    setup_logger()
    idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 03:00"])
    df = pd.DataFrame(
        {
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
        },
        index=idx,
    )
    with capture_logs() as cap:
        ensure_h1(df, policy="pad")
    events = [e for e in cap if e.get("event") == E_DATA_TIME_GAPS]
    assert events
    ev = events[0]
    assert ev["policy"] == "pad"
    assert ev["count"] == 1
    assert ev["gaps_preview"][0]["bars_missing"] == 2
