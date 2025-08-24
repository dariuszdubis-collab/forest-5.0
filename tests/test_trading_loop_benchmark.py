import pytest

pytest.skip("legacy trading loop removed", allow_module_level=True)


def _setup():
    n = 5000
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 1.12 + 0.0001 * np.arange(n)
    df = pd.DataFrame(
        {"open": base, "high": base + 0.0002, "low": base - 0.0002, "close": base},
        index=idx,
    )
    df.index.name = "time"

    settings = BacktestSettings(
        strategy=StrategySettings(name="ema_cross", fast=12, slow=26, use_rsi=False),
        risk=RiskSettings(
            initial_capital=100_000.0,
            risk_per_trade=0.01,
            fee_perc=0.0,
            slippage_perc=0.0,
        ),
        atr_period=1,
        atr_multiple=2.0,
    )

    df = _validate_data(df, price_col="close")
    sig = _generate_signal(df, settings, price_col="close")
    df["atr"] = atr(df["high"], df["low"], df["close"], settings.atr_period)
    return df, sig, settings


def test_trading_loop_benchmark():
    df, sig, settings = _setup()

    def run_old():
        tb = TradeBook()
        rm = RiskManager(**settings.risk.model_dump())
        pos = bootstrap_position(df, sig, rm, tb, settings, "close", settings.atr_multiple)
        _old_trading_loop(df, sig, rm, tb, pos, "close", settings.atr_multiple)

    def run_new():
        tb = TradeBook()
        rm = RiskManager(**settings.risk.model_dump())
        pos = bootstrap_position(df, sig, rm, tb, settings, "close", settings.atr_multiple)
        _trading_loop(df, sig, rm, tb, pos, "close", settings.atr_multiple, settings)

    t_old = timeit.timeit(run_old, number=3)
    t_new = timeit.timeit(run_new, number=3)
    assert t_new <= t_old
