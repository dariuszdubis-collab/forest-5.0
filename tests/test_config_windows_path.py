from pathlib import Path

from forest5.config.loader import _norm_path


def test_windows_literal_path_unchanged(tmp_path: Path) -> None:
    base = tmp_path
    win_path = "C:\\Temp\\model.bin"
    assert _norm_path(base, win_path) == win_path
