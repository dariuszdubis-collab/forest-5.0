from types import SimpleNamespace

from forest5.decision import _normalize_tech_input
from forest5.signals.contract import TechnicalSignal


class Cfg:
    decision = SimpleNamespace(
        tech=SimpleNamespace(default_conf_int=0.2, conf_floor=0.1, conf_cap=0.9),
        weights=SimpleNamespace(tech=1.5),
    )


def test_normalize_int() -> None:
    cfg = Cfg()
    vote = _normalize_tech_input(1, cfg)
    assert vote.direction == 1
    assert vote.weight == 0.2 * 1.5
    assert vote.meta == {"mode": "int"}


def test_normalize_mapping_clamp_floor() -> None:
    cfg = Cfg()
    signal = {"action": "SELL", "technical_score": -2.5, "confidence_tech": 0.05}
    vote = _normalize_tech_input(signal, cfg)
    assert vote.direction == -1
    assert vote.weight == 0.1 * 1.5
    assert vote.score == -2.5
    assert vote.meta["mode"] == "mapping"


def test_normalize_dataclass_clamp_cap() -> None:
    cfg = Cfg()
    sig = TechnicalSignal(action="BUY", technical_score=3.0, confidence_tech=0.95)
    vote = _normalize_tech_input(sig, cfg)
    assert vote.direction == 1
    assert vote.weight == 0.9
    assert vote.score == 3.0
    assert vote.meta["mode"] == "dataclass"


def test_normalize_unknown() -> None:
    cfg = Cfg()
    vote = _normalize_tech_input(object(), cfg)
    assert vote.direction == 0
    assert vote.weight == 0.0
    assert vote.meta["mode"] == "unknown"
