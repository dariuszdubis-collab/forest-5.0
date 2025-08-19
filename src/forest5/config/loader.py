from __future__ import annotations

from pathlib import Path
import os
import re
import yaml

from typing import Type


WINDOWS_PATH_LITERAL_RE = re.compile(r"^[A-Za-z]:\\")


def _is_windows_path_literal(path: str | None) -> bool:
    if path is None:
        return False
    return WINDOWS_PATH_LITERAL_RE.match(path) is not None


def _expand_env_user(v: str) -> str:
    return os.path.expanduser(os.path.expandvars(v))


def _norm_path(v: str | None) -> str | None:
    if v is None:
        return None
    if v == "":
        return ""
    p = Path(_expand_env_user(v))
    return str(p.resolve(strict=False))


def _pydantic_validate(model_cls: Type, data: dict):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)  # type: ignore[call-arg]
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(data)  # type: ignore[attr-defined]
    return model_cls(**data)


def load_live_settings(path: str | Path):
    from ..config_live import LiveSettings

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}

    broker = data.get("broker")
    if isinstance(broker, dict):
        bridge_dir = broker.get("bridge_dir")
        if not _is_windows_path_literal(bridge_dir):
            broker["bridge_dir"] = _norm_path(bridge_dir)
        data["broker"] = broker

    ai = data.get("ai")
    if isinstance(ai, dict):
        ctx = _norm_path(ai.get("context_file"))
        ai["context_file"] = "" if ctx is None else ctx
        data["ai"] = ai

    time = data.get("time")
    if isinstance(time, dict):
        model = time.get("model")
        if isinstance(model, dict):
            mpath = _norm_path(model.get("path"))
            model["path"] = "" if mpath is None else mpath
            time["model"] = model
        data["time"] = time

    if hasattr(LiveSettings, "from_dict"):
        return LiveSettings.from_dict(data)
    return _pydantic_validate(LiveSettings, data)
