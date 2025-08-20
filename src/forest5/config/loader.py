from __future__ import annotations

from pathlib import Path
import os
import re
import yaml

from typing import Any, TYPE_CHECKING, Type

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config_live import LiveSettings


WINDOWS_LITERAL_RE = re.compile(r"^(?:[A-Za-z]:\\|\\\\\\?\\)")


def _is_win_literal(p: str | None) -> bool:
    if not p:
        return False
    return WINDOWS_LITERAL_RE.match(p) is not None


def _expand_env_user(s: str) -> str:
    return os.path.expanduser(os.path.expandvars(s))


def _resolve_from_yaml(base: Path, p: str) -> Path:
    p = _expand_env_user(p)
    if _is_win_literal(p):
        return Path(p)
    q = Path(p)
    if not q.is_absolute():
        q = base / q
    return q.resolve(strict=False)


def _norm_path(base_dir: Path, v: str | None) -> str | None:
    if v in (None, ""):
        return v
    return str(_resolve_from_yaml(base_dir, v))


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
        b_raw = broker.get("bridge_dir")
        if b_raw:
            broker["bridge_dir"] = _resolve_from_yaml(cfg_dir, b_raw)
        else:
            env_bridge = os.getenv("FOREST_MT4_BRIDGE_DIR")
            broker["bridge_dir"] = (
                _resolve_from_yaml(cfg_dir, env_bridge) if env_bridge else None
            )
        data["broker"] = broker

    ai = data.get("ai")
    if isinstance(ai, dict):
        a_raw = ai.get("context_file")
        if a_raw:
            ai["context_file"] = str(_resolve_from_yaml(cfg_dir, a_raw))
        else:
            ai["context_file"] = ""
        data["ai"] = ai

    time = data.get("time")
    if isinstance(time, dict):
        model = time.get("model")
        if isinstance(model, dict):
            m_raw = model.get("path")
            if m_raw:
                model["path"] = _resolve_from_yaml(cfg_dir, m_raw)
            else:
                model["path"] = ""
            time["model"] = model
        data["time"] = time

    if hasattr(LiveSettings, "from_dict"):
        return LiveSettings.from_dict(data)
    return _pydantic_validate(LiveSettings, data)
