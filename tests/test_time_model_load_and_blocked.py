import pandas as pd
from forest5.backtest.engine import run_backtest
from forest5.config import BacktestSettings


def _make_df(periods: int = 2) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="h")
    return pd.DataFrame(
        {"open": [1.0] * periods, "high": [1.0] * periods, "low": [1.0] * periods, "close": [1.0] * periods},
        index=idx,
    )


def test_missing_time_model_does_not_crash(tmp_path, capsys):
    df = _make_df()
    settings = BacktestSettings()
    settings.time.model.enabled = True
    settings.time.model.path = tmp_path / "missing.json"

    run_backtest(df, settings)

    captured = capsys.readouterr().out
    assert "time_model_load_failed" in captured


def test_blocked_candle_logs(capsys):
    df = _make_df(3)
    settings = BacktestSettings()
    settings.time.blocked_hours = [df.index[1].hour]

    run_backtest(df, settings)

    captured = capsys.readouterr().out
    assert "time_blocked" in captured
