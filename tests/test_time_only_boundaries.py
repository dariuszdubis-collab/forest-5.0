from datetime import datetime

from forest5.time_only import TimeOnlyModel


def test_boundaries_sell_buy() -> None:
    model = TimeOnlyModel(quantile_gates={0: (1.0, 2.0)}, q_low=0.1, q_high=0.9)
    ts = datetime(2024, 1, 1, 0, 0)
    assert model.decide(ts, 1.0) == "SELL"
    assert model.decide(ts, 2.0) == "BUY"
