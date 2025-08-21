import pytest

from forest5.cli import main


def _run_and_capture(args, capsys):
    with pytest.raises(SystemExit):
        main(args)
    out = capsys.readouterr().out.lower()
    assert "usage:" in out
    return out


def test_top_level_help(capsys):
    _run_and_capture(["-h"], capsys)


def test_backtest_help(capsys):
    _run_and_capture(["backtest", "--help"], capsys)


def test_live_help(capsys):
    _run_and_capture(["live", "--help"], capsys)
