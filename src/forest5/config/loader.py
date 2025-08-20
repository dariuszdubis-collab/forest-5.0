from __future__ import annotations

from pathlib import Path
import os
import re
import yaml

from typing import Any, TYPE_CHECKING, Type

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config_live import LiveSettings


WINDOWS_PATH_LITERAL_RE = re.compile(r"^[A-Za-z]:\\")


def _is_windows_path_literal(path: str | None) -> bool:
    if path is None:
        return False
    return WINDOWS_PATH_LITERAL_RE.match(path) is not None


def _expand_env_user(v: str) -> str:
    return os.path.expanduser(os.path.expandvars(v))


def _norm_path(base_dir: Path, v: str | None) -> str | None:
    if v is None:
        return None
    if v == "":
        return ""
    if _is_windows_path_literal(v):
        return _expand_env_user(v)
    p = Path(_expand_env_user(v))
    if not p.is_absolute():
        p = base_dir / p
    return str(p.resolve(strict=False))


def _pydantic_validate(model_cls: Type, data: dict) -> Any:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(data)
    return model_cls(**data)


def load_live_settings(path: str | Path) -> "LiveSettings":
    from ..config_live import LiveSettings

    p = Path(path)
    cfg_dir = p.resolve().parent
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}

    broker = data.get("broker")
    if isinstance(broker, dict):
        broker["bridge_dir"] = _norm_path(cfg_dir, broker.get("bridge_dir"))
        data["broker"] = broker

    ai = data.get("ai")
    if isinstance(ai, dict):
        ctx = _norm_path(cfg_dir, ai.get("context_file"))
        ai["context_file"] = "" if ctx is None else ctx
        data["ai"] = ai

    time = data.get("time")
    if isinstance(time, dict):
        model = time.get("model")
        if isinstance(model, dict):
            mpath = _norm_path(cfg_dir, model.get("path"))
            model["path"] = "" if mpath is None else mpath
            time["model"] = model
        data["time"] = time

    if hasattr(LiveSettings, "from_dict"):
        return LiveSettings.from_dict(data)
    return _pydantic_validate(LiveSettings, data)
