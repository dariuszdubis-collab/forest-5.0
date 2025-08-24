import pytest
from forest5.signals.compat import contract_to_int
from forest5.signals.contract import TechnicalSignal


@pytest.mark.parametrize("action,expected", [("BUY", 1), ("SELL", -1), ("KEEP", 0)])
def test_contract_to_int_handles_strings(action: str, expected: int) -> None:
    sig = TechnicalSignal(action=action)
    mapping = {"action": action}
    assert contract_to_int(action) == expected
    assert contract_to_int(mapping) == expected
    assert contract_to_int(sig) == expected
