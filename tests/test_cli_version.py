import pytest

from forest5.cli import build_parser


def test_cli_version_prints_and_exits(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])  # argparse prints and exits
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "forest5" in out.lower()
