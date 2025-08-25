import pandas as pd
import pytest

from forest5.signals.patterns import engulfing, pinbar, stars, registry
from forest5.config.strategy import PatternSettings


def test_engulfing_detects_positive():
    df = pd.DataFrame(
        [
            {"open": 10, "high": 10.5, "low": 9.5, "close": 9.6},
            {"open": 9.4, "high": 10.7, "low": 9.3, "close": 10.8},
        ]
    )
    res = engulfing.detect(df, atr=1.0)
    assert res and res["type"] == "bullish_engulfing"
    assert res["hi"] == 10.7 and res["lo"] == 9.3
    assert res["score"] > 0


def test_engulfing_negative():
    df = pd.DataFrame(
        [
            {"open": 10, "high": 10.5, "low": 9.5, "close": 9.6},
            {"open": 9.8, "high": 10.1, "low": 9.3, "close": 9.9},
        ]
    )
    assert engulfing.detect(df, atr=1.0) is None


def test_pinbar_detects_positive():
    df = pd.DataFrame([{"open": 11, "high": 11.2, "low": 9, "close": 11.1}])
    res = pinbar.detect(df, atr=1.0)
    assert res and res["type"] == "bullish_pinbar"
    assert res["hi"] == 11.2 and res["lo"] == 9
    assert res["score"] > 0


def test_pinbar_negative():
    df = pd.DataFrame([{"open": 10, "high": 10.5, "low": 9.8, "close": 10.2}])
    assert pinbar.detect(df, atr=1.0) is None


def test_stars_detects_positive():
    df = pd.DataFrame(
        [
            {"open": 10, "high": 10.2, "low": 9, "close": 9.2},
            {"open": 9.1, "high": 9.3, "low": 9.0, "close": 9.15},
            {"open": 9.2, "high": 10.5, "low": 9.1, "close": 10.4},
        ]
    )
    res = stars.detect(df, atr=1.0)
    assert res and res["type"] == "morning_star"
    assert res["hi"] == 10.5 and res["lo"] == 9
    assert res["score"] > 0


def test_stars_negative():
    df = pd.DataFrame(
        [
            {"open": 9, "high": 9.5, "low": 8.5, "close": 9.4},
            {"open": 9.6, "high": 10, "low": 9.4, "close": 9.9},
            {"open": 10, "high": 10.5, "low": 9.8, "close": 10.2},
        ]
    )
    assert stars.detect(df, atr=1.0) is None


def test_registry_best_pattern(monkeypatch):
    def fake_a(df, atr):
        return {"type": "a", "hi": 1, "lo": 0, "score": 1}

    def fake_b(df, atr):
        return {"type": "b", "hi": 1, "lo": 0, "score": 2}

    monkeypatch.setitem(registry.PATTERN_DETECTORS, "engulfing", fake_a)
    monkeypatch.setitem(registry.PATTERN_DETECTORS, "pinbar", fake_b)

    res = registry.best_pattern(pd.DataFrame(), 1.0, {"engulfing": True, "pinbar": True})
    assert res["type"] == "b"

    res = registry.best_pattern(pd.DataFrame(), 1.0, {"engulfing": True, "pinbar": False})
    assert res["type"] == "a"


def test_pattern_settings_defaults():
    cfg = PatternSettings()
    assert cfg.enabled is True
    assert cfg.min_strength == pytest.approx(0.6)
    assert cfg.boost_conf == pytest.approx(0.20)
    assert cfg.boost_score == pytest.approx(0.20)
    assert cfg.gate is False
