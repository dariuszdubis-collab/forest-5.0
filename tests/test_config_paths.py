from __future__ import annotations

from pathlib import Path
import textwrap

from forest5.config.loader import load_live_settings, _is_win_literal


def test_path_resolution_and_env_expansion(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "ctx.txt").write_text("ctx", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    model_dir = tmp_path / "cfg"
    model_dir.mkdir()
    (model_dir / "model.bin").write_text("m", encoding="utf-8")
    monkeypatch.setenv("MODEL_FILE", "model.bin")

    bridge = tmp_path / "bridge"
    bridge.mkdir()
    monkeypatch.setenv("FOREST_MT4_BRIDGE_DIR", str(bridge))

    cfg = model_dir / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            broker:
              type: mt4
              bridge_dir: ""
              symbol: EURUSD
              volume: 1.0
            ai:
              context_file: "~/ctx.txt"
            time:
              model:
                path: "${MODEL_FILE}"
            """
        ),
        encoding="utf-8",
    )

    s = load_live_settings(cfg)

    assert s.ai.context_file == str((home / "ctx.txt").resolve())
    assert s.time.model.path == (model_dir / "model.bin").resolve()
    assert s.broker.bridge_dir == bridge.resolve()
    assert s.time.model.path.is_absolute()
    assert s.broker.bridge_dir.is_absolute()
    assert Path(s.ai.context_file).is_absolute()


def test_windows_literal_preserved(tmp_path: Path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            broker:
              type: mt4
              bridge_dir: 'C:\\Temp\\bridge'
              symbol: EURUSD
              volume: 1.0
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
    assert _is_win_literal(s.ai.context_file)
