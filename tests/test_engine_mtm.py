import numpy as np
import pandas as pd

from forest5.config import BacktestSettings
from forest5.backtest.engine import run_backtest
from forest5.examples.synthetic import generate_ohlc


def test_equity_curve_starts_at_initial_and_never_drops_to_price(monkeypatch):
    # Wymuszamy sygnał BUY na pierwszym barze, dalej brak zmian
    import forest5.backtest.engine as eng

    def fake_signal(df, settings, price_col="close", compat_int=False):
        # 1 na pierwszym barze => otwarcie longa, dalej 0
        arr = np.zeros(len(df), dtype=int)
        arr[0] = 1
        return pd.Series(arr, index=df.index, dtype=int)

    monkeypatch.setattr(eng, "compute_signal", fake_signal)

    df = generate_ohlc(periods=10, start_price=100.0, freq="D")
    res = run_backtest(df, BacktestSettings())

    # 1) pierwsza próbka equity powinna być initial_capital (nie cena)
    assert res.equity_curve.iloc[0] == 100_000.0

    # 2) krzywa equity nie może "spadać" w okolice ~100 (cena),
    #    co się działo, gdy zapisywaliśmy price zamiast equity
    assert res.equity_curve.min() > 1000.0  # solidny bufor


def test_mtm_drawdown_triggers_on_real_equity(monkeypatch):
    # Tworzymy "spadkowy" rynek i wymuszamy długą pozycję od pierwszego bara.
    import forest5.backtest.engine as eng

    def fake_signal(df, settings, price_col="close", compat_int=False):
        arr = np.zeros(len(df), dtype=int)
        arr[0] = 1  # otwarcie longa
        return pd.Series(arr, index=df.index, dtype=int)

    monkeypatch.setattr(eng, "compute_signal", fake_signal)

    # Linowy spadek ceny -> DD powinien wzrosnąć powyżej progu strategii (~0.3)
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    close = np.linspace(100.0, 60.0, len(idx))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        },
        index=idx,
    )

    res = run_backtest(df, BacktestSettings())
    assert res.max_dd >= 0.20
