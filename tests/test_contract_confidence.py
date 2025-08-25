from types import SimpleNamespace

import pytest

from forest5.decision import _normalize_tech_input
from forest5.signals.contract import TechnicalSignal


class DummyCfg:
    decision = SimpleNamespace(
        tech=SimpleNamespace(default_conf_int=1.0, conf_floor=0.0, conf_cap=1.0),
        weights=SimpleNamespace(tech=1.0),
    )


def test_contract_confidence_affects_weight() -> None:
    cfg = DummyCfg()
    low_conf = TechnicalSignal(action="BUY", technical_score=1.0, confidence_tech=0.2)
    high_conf = TechnicalSignal(action="BUY", technical_score=1.0, confidence_tech=0.8)

    low_vote = _normalize_tech_input(low_conf, cfg)
    high_vote = _normalize_tech_input(high_conf, cfg)

    assert low_vote.weight < high_vote.weight


def test_int_legacy_path_unchanged() -> None:
    cfg = DummyCfg()
    vote = _normalize_tech_input(1, cfg)

    assert vote.direction == 1
    assert vote.weight == 1.0
    assert vote.meta == {"mode": "int"}


def test_weight_clamped_to_cap() -> None:
    cfg = DummyCfg()
    cfg.decision.tech.conf_cap = 0.7
    cfg.decision.weights.tech = 2.0
    sig = TechnicalSignal(action="BUY", technical_score=1.0, confidence_tech=1.0)
    vote = _normalize_tech_input(sig, cfg)

    assert vote.weight == pytest.approx(0.7)
