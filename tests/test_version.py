from pathlib import Path

import tomllib

from forest5 import __version__


def test_version_matches_pyproject() -> None:
    pyproject_version = tomllib.loads(
        Path("pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["poetry"]["version"]
    assert __version__ == pyproject_version
