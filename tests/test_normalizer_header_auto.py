import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from normalize_fxdata import normalize_file


def test_normalizer_header_auto(tmp_path):
    csv = tmp_path / "raw.csv"
    with open(csv, "w") as f:
        f.write("Time,Open,High,Low,Close,Volume\n")
        f.write("2024-01-01 00:00,1,2,0,1.5,100\n")
        f.write("2024-01-01 01:00,1,2,0,1.5,100\n")
        f.write("2024-01-01 02:00,1,2,0,1.5,100\n")
    df = normalize_file(csv, schema="auto", tz="UTC", floor_to_hour=True)
    assert df.index.tz is not None
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert pd.infer_freq(df.index).lower() == "h"
