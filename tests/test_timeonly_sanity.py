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

    model = time_only.TimeOnlyModel(prob_tables={0: {"BUY": 1.0}}, quantiles=[0.25, 0.75])
    path = tmp_path / "model_time.json"
    model.save(path)
    loaded = time_only.TimeOnlyModel.load(path)

    ts = datetime(2024, 1, 1, 0)
    assert loaded.decide(ts)["decision"] in {"BUY", "SELL", "WAIT"}
