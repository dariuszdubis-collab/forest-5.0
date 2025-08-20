from pathlib import Path

from forest5.live.live_runner import run_live
from forest5.live.settings import (
    LiveSettings,
    BrokerSettings,
    DecisionSettings,
    AISettings,
    TimeSettings,
    RiskSettings,
)


def _mk_bridge(tmpdir: Path) -> Path:
    bridge = tmpdir / "forest_bridge"
    for sub in ("ticks", "state", "commands", "results"):
        (bridge / sub).mkdir(parents=True, exist_ok=True)
    (bridge / "ticks" / "tick.json").write_text(
        '{"symbol":"EURUSD","bid":1.0,"ask":1.0,"time":0}',
        encoding="utf-8",
    )
    (bridge / "state" / "account.json").write_text(
        '{"equity":10000}', encoding="utf-8"
    )
    (bridge / "state" / "position_EURUSD.json").write_text(
        '{"qty":0}', encoding="utf-8"
    )
    return bridge


def test_run_live_paper(tmp_path: Path):
    bridge = _mk_bridge(tmp_path)
    s = LiveSettings(
        broker=BrokerSettings(
            type="paper", bridge_dir=str(bridge), symbol="EURUSD", volume=0.01
        ),
        decision=DecisionSettings(min_confluence=1),
        ai=AISettings(enabled=False, model="gpt-4o-mini", max_tokens=64, context_file=None),
        time=TimeSettings(blocked_hours=[], blocked_weekdays=[]),
        risk=RiskSettings(max_drawdown=0.5, on_drawdown={"action": "halt"}),
    )
    run_live(s, max_steps=2)
