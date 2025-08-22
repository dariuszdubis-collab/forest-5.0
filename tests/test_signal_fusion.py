import pytest

from forest5.signals.fusion import fuse_signals


@pytest.mark.parametrize(
    "tech_sig,time_sig,ai_vote,expected,reason",
    [
        # time model waits -> short-circuit regardless of AI vote
        (1, "WAIT", 1, 0, "time_wait"),
        (1, "WAIT", -1, 0, "time_wait"),
    ],
)
def test_time_wait_short_circuit(tech_sig, time_sig, ai_vote, expected, reason):
    fused, why = fuse_signals(tech_sig, time_sig, ai=ai_vote)
    assert fused == expected
    assert why == reason

