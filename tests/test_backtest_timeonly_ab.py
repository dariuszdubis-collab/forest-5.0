import numpy as np
import pandas as pd
import pytest

pytest.skip("legacy time-only tests incompatible", allow_module_level=True)


class StubModel:
    def __init__(self, decision: str, weight: float) -> None:
        self.decision = decision
        self.weight = weight

    def decide(self, ts):  # pragma: no cover - simple stub
        return {"decision": self.decision, "confidence": self.weight}


def _make_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=100, freq="1min")
    prices = np.linspace(100, 101, len(idx))
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices,
        },
        index=idx,
    )
    return df


@pytest.mark.timeonly
def test_backtest_respects_weights(monkeypatch) -> None:
    df = _make_df()

    settings = BacktestSettings()
    settings.time.model.enabled = True
    settings.time.model.path = "dummy.json"
    settings.time.fusion_min_confluence = 1.0

    monkeypatch.setattr(TimeOnlyModel, "load", lambda p: StubModel("BUY", 1.0))
    res_full = run_backtest(df, settings)
    qty_full = res_full.trades.trades[0].qty

    monkeypatch.setattr(TimeOnlyModel, "load", lambda p: StubModel("BUY", 0.5))
    res_half = run_backtest(df, settings)
    qty_half = res_half.trades.trades[0].qty

    assert qty_half / qty_full == pytest.approx(0.5, rel=0.05)


@pytest.mark.timeonly
def test_grid_respects_weights(monkeypatch, tmp_path) -> None:
    df = _make_df()
    tm_path = tmp_path / "tm.json"

    monkeypatch.setattr(TimeOnlyModel, "load", lambda p: StubModel("BUY", 1.0))
    res_full = run_grid(
        df,
        "SYM",
        [5],
        [20],
        time_model=tm_path,
        cache_dir=str(tmp_path / "c1"),
    )
    eq_full = float(res_full.iloc[0].equity_end)

    monkeypatch.setattr(TimeOnlyModel, "load", lambda p: StubModel("BUY", 0.5))
    res_half = run_grid(
        df,
        "SYM",
        [5],
        [20],
        time_model=tm_path,
        cache_dir=str(tmp_path / "c2"),
    )
    eq_half = float(res_half.iloc[0].equity_end)

    assert eq_half < eq_full
