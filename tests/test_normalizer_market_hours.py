import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from normalize_fxdata import normalize_file


def test_normalizer_market_hours(tmp_path):
    csv = tmp_path / "raw.csv"
    with open(csv, "w") as f:
        f.write("time,open,high,low,close,volume\n")
        f.write("2024-01-05 23:30,1,2,0,1.5,100\n")
        f.write("2024-01-07 00:30,1,2,0,1.5,100\n")
    df = normalize_file(csv, schema="auto", tz="UTC", floor_to_hour=True, weekend="pad")
    assert pd.infer_freq(df.index).lower() == "h"
    assert df.index[0] == pd.Timestamp("2024-01-05 23:00", tz="UTC")
