import pandas as pd
from forest5.backtest.engine import run_backtest
from forest5.config import BacktestSettings, StrategySettings


def _asdict_trade(tr):
    # TradeBook zwykle trzyma dataclass lub dict – obsłużymy oba warianty.
    if hasattr(tr, "__dict__"):
        return tr.__dict__
    return dict(tr)


def test_no_overbuy_fx():
    # Zmieniamy poziom po 30 barach, żeby wymusić BUY (cross).
    close = [1.10] * 30 + [1.30] * 30
    high = [c + 0.0002 for c in close]
    low = [c - 0.0002 for c in close]
    idx = pd.date_range("2024-01-01", periods=len(close), freq="h")
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)

    s = BacktestSettings(strategy=StrategySettings(name="ema_cross", fast=1, slow=3, use_rsi=False))
    res = run_backtest(df, s)

    cap = s.risk.initial_capital
    for tr in res.trades.trades:
        d = _asdict_trade(tr)
        assert {"entry", "sl", "tp", "reason_close", "setup_id", "pattern"}.issubset(d)
        if d.get("side") == "BUY":
            assert d["price_open"] * d["qty"] <= cap + 1e-6, "BUY przekracza dostępny kapitał!"
