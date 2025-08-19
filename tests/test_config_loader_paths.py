from pathlib import Path
import os
import sys
import types
import importlib.util
import pytest

numpy_stub = types.ModuleType("numpy")
sys.modules.setdefault("numpy", numpy_stub)
pandas_stub = types.ModuleType("pandas")
sys.modules.setdefault("pandas", pandas_stub)

pydantic_stub = types.ModuleType("pydantic")
class BaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def Field(*a, **k):
    return None

def field_validator(*a, **k):
    def deco(f):
        return f
    return deco
pydantic_stub.BaseModel = BaseModel
pydantic_stub.Field = Field
pydantic_stub.field_validator = field_validator
sys.modules.setdefault("pydantic", pydantic_stub)


def _safe_load(text: str) -> dict:
    levels: dict[int, dict] = {0: {}}
    for line in text.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        value = value.strip()
        if value == "":
            parent = levels[indent]
            child: dict = {}
            parent[key] = child
            levels[indent + 2] = child
        else:
            parent = levels[indent]
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            parent[key] = None if value == "null" else value
    return levels[0]


yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = _safe_load
sys.modules.setdefault("yaml", yaml_stub)

repo_root = Path(__file__).resolve().parents[1]
package_root = repo_root / "src/forest5"
forest5_pkg = types.ModuleType("forest5")
forest5_pkg.__path__ = [str(package_root)]
sys.modules.setdefault("forest5", forest5_pkg)

spec_live = importlib.util.spec_from_file_location(
    "forest5.config_live", package_root / "config_live.py"
)
config_live = importlib.util.module_from_spec(spec_live)
sys.modules["forest5.config_live"] = config_live
spec_live.loader.exec_module(config_live)  # type: ignore[union-attr]

spec_loader = importlib.util.spec_from_file_location(
    "forest5.config.loader", package_root / "config" / "loader.py"
)
loader = importlib.util.module_from_spec(spec_loader)
sys.modules["forest5.config.loader"] = loader
spec_loader.loader.exec_module(loader)  # type: ignore[union-attr]
load_live_settings = loader.load_live_settings


def test_paths_relative_and_env_and_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("hi", encoding="utf-8")
    monkeypatch.setenv("CTX", str(ctx))

    monkeypatch.setenv("HOME", str(tmp_path))
    model = tmp_path / "model.bin"
    model.write_text("m", encoding="utf-8")

    bridge = tmp_path / "bridge"
    bridge.mkdir()

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "broker:\n"
        "  type: mt4\n"
        "  bridge_dir: bridge\n"
        "ai:\n"
        "  context_file: ${CTX}\n"
        "time:\n"
        "  model:\n"
        "    path: ~/model.bin\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    s = load_live_settings(cfg)

    assert s.ai.context_file == str(ctx)
    assert os.fspath(s.time.model.path) == str(model)
    assert os.fspath(s.broker.bridge_dir) == str(bridge)


def test_windows_literal_bridge_is_preserved(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "broker:\n"
        "  type: mt4\n"
        "  bridge_dir: \"C:\\Users\\bob\\bridge\"\n",
        encoding="utf-8",
    )

    s = load_live_settings(cfg)

    assert os.fspath(s.broker.bridge_dir) == "C:\\Users\\bob\\bridge"
