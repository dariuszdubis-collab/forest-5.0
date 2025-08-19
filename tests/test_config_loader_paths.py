from __future__ import annotations

from pathlib import Path
import textwrap

from forest5.config.loader import load_live_settings


def test_load_live_settings_expands_paths(tmp_path: Path, monkeypatch):
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("hi", encoding="utf-8")
    model = tmp_path / "model.bin"
    model.write_text("m", encoding="utf-8")
    bridge = tmp_path / "bridge"
    bridge.mkdir()

    monkeypatch.setenv("CTX_PATH", str(ctx))
    monkeypatch.setenv("MODEL_PATH", str(model))
    monkeypatch.setenv("BRIDGE_PATH", str(bridge))

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            broker:
              type: mt4
              bridge_dir: "${BRIDGE_PATH}"
            ai:
              context_file: "${CTX_PATH}"
            time:
              model:
                path: "${MODEL_PATH}"
            """
        ),
        encoding="utf-8",
    )

    s = load_live_settings(cfg)

    assert s.ai.context_file == str(ctx)
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

