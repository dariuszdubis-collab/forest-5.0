from forest5.backtest.grid import run_param_grid
from forest5.config import BacktestSettings
from forest5.examples.synthetic import generate_ohlc


def _settings() -> BacktestSettings:
    return BacktestSettings(symbol="SYMB", timeframe="1h", strategy={"name": "ema_cross"})


def _run_and_capture(capsys) -> list[str]:
    df = generate_ohlc(periods=20, start_price=100.0, freq="D")
    params = {"fast": [5, 6], "slow": [10, 12]}
    run_param_grid(df, _settings(), params, jobs=0)
    cap = capsys.readouterr()
    return (cap.out + cap.err).splitlines()


def test_progress_eta_throttled(capsys):
    lines = _run_and_capture(capsys)
    text = "\n".join(lines)
    assert "0/4" in text and "4/4" in text
    assert "eta" in text.lower()
    assert "best" in text.lower()
    assert len(lines) <= 10
