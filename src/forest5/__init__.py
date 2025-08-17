# src/forest5/__init__.py
"""Forest 5.0 – zestaw narzędzi do researchu i backtestów."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib

try:
    __version__ = version("forest5")
except PackageNotFoundError:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    __version__ = tomllib.loads(pyproject.read_text())["tool"]["poetry"]["version"]

from . import time_only

__all__ = ["__version__", "time_only"]
