from pathlib import Path

from forest5.live.live_runner import _read_context


def test_read_context_truncates(tmp_path: Path) -> None:
    p = tmp_path / "ctx.txt"
    p.write_text("abcdefghij", encoding="utf-8")
    assert _read_context(p, 5) == "abcde"


def test_read_context_removes_binary_noise(tmp_path: Path) -> None:
    p = tmp_path / "ctx.txt"
    p.write_bytes(b"hello\xffworld")
    assert _read_context(p, 100) == "helloworld"
