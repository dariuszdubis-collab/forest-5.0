from datetime import datetime
import math
import pytest

from forest5.time_only import TimeOnlyModel


@pytest.mark.parametrize(
    ("q_low", "q_high", "value", "decision"),
    [
        (0.0, 0.9, 0.5, "SELL"),
        (0.1, 1.0, 3.0, "BUY"),
    ],
)
def test_extreme_quantiles_produce_finite_weights(q_low, q_high, value, decision) -> None:
    model = TimeOnlyModel(quantile_gates={0: (1.0, 2.0)}, q_low=q_low, q_high=q_high)
    ts = datetime(2024, 1, 1, 0, 0)
    dec, w = model.decide(ts, value)
    assert dec == decision
    assert math.isfinite(w)
