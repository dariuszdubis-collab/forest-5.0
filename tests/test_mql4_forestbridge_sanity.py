from pathlib import Path

import pytest

SRC = (Path(__file__).resolve().parents[1] / "mt4" / "ForestBridge.mq4").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "token",
    [
        "OnInit",
        "OnTick",
        "OnTimer",
        "ProcessCommands",
        "OrderSend",
        "FileOpen",
        "tick.json",
        "commands",
        "results",
        "state",
    ],
)
def test_forestbridge_contains(token: str) -> None:
    assert token in SRC
