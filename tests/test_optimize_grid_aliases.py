import sys
import pandas as pd
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / "scripts" / "optimize_grid.py"
spec = importlib.util.spec_from_file_location("optimize_grid_module", path)
optimize_grid = importlib.util.module_from_spec(spec)
sys.modules["optimize_grid_module"] = optimize_grid
spec.loader.exec_module(optimize_grid)


def test_optimize_grid_aliases(tmp_path, monkeypatch):
    csv = tmp_path / "d.csv"
    csv.write_text("time,open,high,low,close\n", encoding="utf-8")
    def fake_load_csv(p):
        raise SystemExit(0)
    monkeypatch.setattr(optimize_grid, "_load_csv", fake_load_csv)
    for fast, slow in [("--fast", "--slow"), ("--ema-fast", "--ema-slow")]:
        argv = [
            "optimize_grid.py",
            "--csv",
            str(csv),
            "--symbol",
            "EURUSD",
            fast,
            "5-10",
            slow,
            "20-30",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            optimize_grid.main()
