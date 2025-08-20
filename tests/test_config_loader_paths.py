from __future__ import annotations

from pathlib import Path
import textwrap

from forest5.config.loader import load_live_settings


def test_load_live_settings_expands_paths(tmp_path: Path, monkeypatch):
    bridge = tmp_path / "bridge"
    bridge.mkdir()

    model = tmp_path / "model.bin"
    model.write_text("m", encoding="utf-8")

    monkeypatch.setenv("BRIDGE_PATH", str(bridge))
    monkeypatch.setenv("HOME", str(tmp_path))

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            broker:
              type: mt4
              bridge_dir: "${BRIDGE_PATH}"
            ai:
              context_file: "~/ctx.txt"
            time:
              model:
                path: "model.bin"
            """
        ),
        encoding="utf-8",
    )

    (tmp_path / "ctx.txt").write_text("hi", encoding="utf-8")

    s = load_live_settings(cfg)

    assert s.ai.context_file == str(tmp_path / "ctx.txt")
    assert str(s.time.model.path) == str(model)
    assert str(s.broker.bridge_dir) == str(bridge)


def test_load_live_settings_none_to_empty(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            broker:
              type: mt4
            ai:
              context_file: null
            time:
              model:
                path: null
            """
        ),
        encoding="utf-8",
    )

    s = load_live_settings(cfg)

    assert s.ai.context_file == ""
    assert str(getattr(s.time.model, "path", "")) in {"", "."}


def test_load_live_settings_windows_literal(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            broker:
              type: mt4
              bridge_dir: 'C:\\Temp\\bridge'
            ai:
              context_file: 'C:\\Temp\\ctx.txt'
            time:
              model:
                path: 'C:\\Temp\\model.bin'
            """
        ),
        encoding="utf-8",
    )

    s = load_live_settings(cfg)

    assert s.ai.context_file == "C:\\Temp\\ctx.txt"
    assert str(s.time.model.path) == "C:\\Temp\\model.bin"
    assert str(s.broker.bridge_dir) == "C:\\Temp\\bridge"

