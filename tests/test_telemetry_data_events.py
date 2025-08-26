import json
import logging
import pandas as pd

from forest5.cli import main
from forest5.utils.log import E_DATA_CSV_SCHEMA, E_DATA_TIME_GAPS, setup_logger


def test_cli_inspect_emits_schema_event(tmp_path, caplog, capsys):
    csv = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=2, freq="h"),
            "open": [1, 1.1],
            "high": [1, 1.1],
            "low": [1, 1.1],
            "close": [1, 1.1],
        }
    )
    df.to_csv(csv, index=False)

    setup_logger()
    caplog.set_level(logging.INFO)
    main(["data", "inspect", "--csv", str(csv)])
    out = caplog.text + capsys.readouterr().out
    events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    ev = next(e for e in events if e.get("event") == E_DATA_CSV_SCHEMA)
    assert ev["path"] == str(csv)
    assert ev["rows"] == 2


def test_cli_pad_h1_emits_gap_event(tmp_path, caplog, capsys):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    pad = in_dir / "pad.csv"
    pad.write_text(
        "time,open,high,low,close\n" "2020-01-01 00:00,1,1,1,1\n" "2020-01-01 02:00,1,1,1,1\n",
        encoding="utf-8",
    )

    setup_logger()
    caplog.set_level(logging.INFO)
    main(["data", "pad-h1", "--input-dir", str(in_dir), "--out-dir", str(out_dir)])
    out = caplog.text + capsys.readouterr().out
    events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    ev = next(e for e in events if e.get("event") == E_DATA_TIME_GAPS)
    assert ev["count"] == 1
    assert ev["gaps_preview"][0]["bars_missing"] == 1
