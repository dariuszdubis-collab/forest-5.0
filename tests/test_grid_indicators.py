from forest5.backtest.grid import run_grid
from forest5.examples.synthetic import generate_ohlc
from forest5.time_only import TimeOnlyModel


class StubModel:
    def decide(self, ts):  # pragma: no cover - simple stub
        return {"decision": "BUY", "confidence": 1.0}


def test_grid_with_rsi_and_time_model(tmp_path, monkeypatch):
    df = generate_ohlc(periods=40, start_price=100.0, freq="h")
    model_path = tmp_path / "model_time.json"
    model_path.write_text("{}")

    monkeypatch.setattr(TimeOnlyModel, "load", lambda p: StubModel())

    res = run_grid(
        df,
        symbol="SYMB",
        fast_values=[5],
        slow_values=[10],
        atr_period=5,
        atr_multiple=1.5,
        use_rsi=True,
        time_model=model_path,
        cache_dir=str(tmp_path / "cache"),
        n_jobs=1,
    )

    assert not res.empty
    assert {"equity_end", "max_dd", "cagr", "rar"}.issubset(res.columns)
