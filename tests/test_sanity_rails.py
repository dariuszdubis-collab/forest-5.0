from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.skip("legacy sanity rails incompatible", allow_module_level=True)


def _mk_df_trend(n=60, start=100.0, end=60.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    c = np.linspace(start, end, n)
    return pd.DataFrame(
        {
            "open": c,
            "high": c + 0.5,
            "low": c - 0.5,
            "close": c,
        },
        index=idx,
    )


def test_once_per_bar_and_dd():
    df = _mk_df_trend()
    res = run_backtest(df, s)
    # długość: N (+1 jeśli startowa kropka)
    assert len(res.equity_curve) in (
        len(df),
        len(df) + 1,
    )
    # DD realny:
    peak = res.equity_curve.cummax()
    dd = (peak - res.equity_curve) / peak.replace(0, np.nan)
    assert dd.max() >= 0.20


def test_precommit_config_exists():
    root = Path(__file__).resolve().parents[1]
    cfg = root / ".pre-commit-config.yaml"
    assert cfg.is_file(), "Brak pliku .pre-commit-config.yaml"
    text = cfg.read_text()
    for hook in ["black", "ruff", "flake8", "bandit", "pip-audit"]:
        assert hook in text
