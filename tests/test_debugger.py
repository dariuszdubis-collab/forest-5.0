import json
from forest5.utils.debugger import DebugLogger


def test_debug_logger(tmp_path):
    logger = DebugLogger(tmp_path)
    logger.log("foo", value=1)
    logger.log("bar", value=2)
    log_file = tmp_path / "decision_log.jsonl"
    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert lines == [
        {"event": "foo", "value": 1},
        {"event": "bar", "value": 2},
    ]
    logger.close()
