import pytest


@pytest.mark.timeonly
def test_env_sanity_imports() -> None:
    import yaml
    import numpy
    import pandas
    import pydantic
    from forest5 import time_only

    _ = yaml, numpy, pandas, pydantic, time_only


@pytest.mark.timeonly
def test_timeonly_save_load_and_decide_roundtrip(tmp_path) -> None:
    from datetime import datetime
    from forest5 import time_only

    model = time_only.TimeOnlyModel({0: (1.0, 2.0)}, q_low=0.25, q_high=0.75)
    path = tmp_path / "model_time.json"
    model.save(path)
    loaded = time_only.TimeOnlyModel.load(path)

    ts = datetime(2024, 1, 1, 0)
    dec, _ = loaded.decide(ts, 0.5)
    assert dec == "SELL"
    dec, _ = loaded.decide(ts, 2.5)
    assert dec == "BUY"
    dec, _ = loaded.decide(ts, 1.5)
    assert dec == "WAIT"
