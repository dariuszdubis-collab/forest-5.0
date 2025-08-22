from datetime import datetime

from forest5.time_only import TimeOnlyModel


def test_boundaries_sell_buy() -> None:
    model = TimeOnlyModel(prob_tables={0: {"BUY": 1.0}}, quantiles=[0.1, 0.9])
    ts = datetime(2024, 1, 1, 0, 0)
    assert model.decide(ts)["decision"] == "BUY"
