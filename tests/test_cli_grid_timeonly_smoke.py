import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import pandas as pd
import pytest

from forest5 import time_only


script = Path(__file__).resolve().parents[1] / "scripts" / "optimize_grid.py"
pytestmark = pytest.mark.skipif(not script.exists(), reason="optimize_grid.py not found")


@pytest.mark.timeonly
@pytest.mark.grid
@pytest.mark.slow
def test_cli_grid_timeonly_smoke(tmp_path):
    csv_path = tmp_path / "data.csv"
    idx = pd.date_range("2020-01-01", periods=20, freq="h")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": [1.0] * len(idx),
            "high": [1.5] * len(idx),
            "low": [0.5] * len(idx),
            "close": [1.0] * len(idx),
        }
    )
    df.to_csv(csv_path, index=False)

    model_path = tmp_path / "model_time.json"
    train_df = pd.DataFrame({"time": idx, "y": [0.0] * len(idx)})
    time_only.train(train_df).save(model_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1] / "src"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(script),
            "--csv",
            str(csv_path),
            "--fast",
            "5-5",
            "--slow",
            "10-10",
            "--time-model",
            str(model_path),
            "--min-confluence",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0  # nosec B101
