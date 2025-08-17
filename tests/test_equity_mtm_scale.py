import pandas as pd
from forest5.backtest.engine import run_backtest
from forest5.config import BacktestSettings

def test_equity_curve_scale_no_trades():
    # Stała cena -> brak sygnałów -> brak transakcji.
    idx = pd.date_range("2024-01-01", periods=120, freq="h")
    df = pd.DataFrame(
        {"open": 1.234, "high": 1.234, "low": 1.234, "close": 1.234},
        index=idx,
    )
    s = BacktestSettings()  # domyślne ustawienia (ema_cross, brak RSI)
    res = run_backtest(df, s)
    cap = s.risk.initial_capital
    # Jeżeli do equity wpisujesz cenę, test się wywali, bo min ~ 1.234 << 100k.
    assert res.equity_curve.min() >= 0.99 * cap

