import pandas as pd
import pytest

from forest5.signals.h1_ema_rsi_atr import compute_primary_signal_h1
from forest5.signals.setups import SetupRegistry


@pytest.fixture
def base_params():
    return {
        "ema_fast": 2,
        "ema_slow": 3,
        "atr_period": 2,
        "rsi_period": 2,
        "t_sep_atr": 0.5,
        "pullback_atr": 0.5,
        "entry_buffer_atr": 0.1,
        "sl_atr": 0.5,
        "sl_min_atr": 1.0,
        "rr": 2.0,
        "timeframe": "H1",
        "horizon_minutes": 60,
    }


def _make_df():
    return pd.DataFrame(
        [
            {"open": 0, "high": 0, "low": 0, "close": 0},
            {"open": 0, "high": 0, "low": 0, "close": 0},
            {"open": 0, "high": 0, "low": 0, "close": 0},
            {"open": 1, "high": 2, "low": 0, "close": 1},
            {"open": 2, "high": 3, "low": 0, "close": 2},
        ]
    )


def test_h1_signal_triggers_buy_with_targets(monkeypatch, base_params):
    df = _make_df()

    def fake_ema(close, period):
        n = len(close)
        if period == base_params["ema_fast"]:
            vals = [1] * (n - 1) + [2]
        else:
            vals = [0] * n
        return pd.Series(vals, index=close.index)

    def fake_atr(high, low, close, period):
        return pd.Series([1.0] * len(close), index=close.index)

    def fake_rsi(close, period):
        n = len(close)
        vals = [40] * (n - 2) + [40, 60]
        return pd.Series(vals, index=close.index)

    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.ema", fake_ema)
    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.atr", fake_atr)
    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.rsi", fake_rsi)

    reg = SetupRegistry()
    compute_primary_signal_h1(df, base_params, registry=reg)

    # breakout on next bar
    df2 = pd.concat(
        [df, pd.DataFrame([{"open": 2, "high": 3.2, "low": 2.0, "close": 2.5}])],
        ignore_index=True,
    )
    res = compute_primary_signal_h1(df2, base_params, registry=reg)
    assert res.action == "BUY"
    assert pytest.approx(res.entry, rel=1e-6) == 3.1
    assert pytest.approx(res.sl, rel=1e-6) == 2.1
    assert pytest.approx(res.tp, rel=1e-6) == 5.1


def test_h1_signal_requires_rsi_cross(monkeypatch, base_params):
    df = _make_df()

    def fake_ema(close, period):
        n = len(close)
        if period == base_params["ema_fast"]:
            vals = [1] * (n - 1) + [2]
        else:
            vals = [0] * n
        return pd.Series(vals, index=close.index)

    def fake_atr(high, low, close, period):
        return pd.Series([1.0] * len(close), index=close.index)

    def fake_rsi(close, period):
        n = len(close)
        vals = [40] * (n - 2) + [40, 45]
        return pd.Series(vals, index=close.index)

    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.ema", fake_ema)
    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.atr", fake_atr)
    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.rsi", fake_rsi)

    reg = SetupRegistry()
    compute_primary_signal_h1(df, base_params, registry=reg)

    df2 = pd.concat(
        [df, pd.DataFrame([{"open": 2, "high": 3.5, "low": 2.0, "close": 2.5}])],
        ignore_index=True,
    )
    res = compute_primary_signal_h1(df2, base_params, registry=reg)
    assert res.action == "KEEP"


def test_h1_signal_patterns_boost(monkeypatch, base_params):
    params = {
        **base_params,
        "patterns": {
            "enabled": True,
            "min_strength": 0.1,
            "boost_conf": 0.2,
            "boost_score": 0.5,
        },
    }

    df = _make_df()

    def fake_ema(close, period):
        n = len(close)
        if period == base_params["ema_fast"]:
            vals = [1] * (n - 1) + [2]
        else:
            vals = [0] * n
        return pd.Series(vals, index=close.index)

    def fake_atr(high, low, close, period):
        return pd.Series([1.0] * len(close), index=close.index)

    def fake_rsi(close, period):
        n = len(close)
        vals = [40] * (n - 2) + [40, 60]
        return pd.Series(vals, index=close.index)

    def fake_best_pattern(df, atr, cfg):
        return {"name": "pinbar", "strength": 0.3}

    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.ema", fake_ema)
    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.atr", fake_atr)
    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.rsi", fake_rsi)
    monkeypatch.setattr(
        "forest5.signals.h1_ema_rsi_atr.patterns.registry.best_pattern", fake_best_pattern
    )

    reg = SetupRegistry()
    compute_primary_signal_h1(df, params, registry=reg)

    df2 = pd.concat(
        [df, pd.DataFrame([{"open": 2, "high": 3.2, "low": 2.0, "close": 2.5}])],
        ignore_index=True,
    )
    res = compute_primary_signal_h1(df2, params, registry=reg)
    assert res.technical_score > 1.0
    assert any(isinstance(d, dict) and d.get("pattern") == "pinbar" for d in res.drivers)
    assert res.meta.get("pattern") == "pinbar"


def test_h1_signal_patterns_gate(monkeypatch, base_params):
    params = {**base_params, "patterns": {"enabled": True, "gate": True}}

    df = _make_df()

    def fake_ema(close, period):
        n = len(close)
        if period == base_params["ema_fast"]:
            vals = [1] * (n - 1) + [2]
        else:
            vals = [0] * n
        return pd.Series(vals, index=close.index)

    def fake_atr(high, low, close, period):
        return pd.Series([1.0] * len(close), index=close.index)

    def fake_rsi(close, period):
        n = len(close)
        vals = [40] * (n - 2) + [40, 60]
        return pd.Series(vals, index=close.index)

    def no_pattern(df, atr, cfg):
        return None

    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.ema", fake_ema)
    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.atr", fake_atr)
    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.rsi", fake_rsi)
    monkeypatch.setattr("forest5.signals.h1_ema_rsi_atr.patterns.registry.best_pattern", no_pattern)

    reg = SetupRegistry()
    compute_primary_signal_h1(df, params, registry=reg)

    df2 = pd.concat(
        [df, pd.DataFrame([{"open": 2, "high": 3.2, "low": 2.0, "close": 2.5}])],
        ignore_index=True,
    )
    res = compute_primary_signal_h1(df2, params, registry=reg)
    assert res.action == "KEEP"
    assert not reg._setups
