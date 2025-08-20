from pathlib import Path
from forest5.config import BacktestSettings


def test_config_from_yaml(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "symbol: EURUSD\n"
        "timeframe: H\n"
        "strategy:\n"
        "  name: ema_cross\n"
        "  fast: 10\n"
        "  slow: 30\n"
        "risk:\n"
        "  initial_capital: 50000.0\n"
        "  risk_per_trade: 0.02\n",
        encoding="utf-8",
    )
    s = BacktestSettings.from_file(p)
    assert s.symbol == "EURUSD"
    assert s.timeframe == "1h"
    assert s.strategy.fast == 10
    assert s.risk.initial_capital == 50_000.0
    assert s.risk.on_drawdown.action == "halt"  # nosec B101
