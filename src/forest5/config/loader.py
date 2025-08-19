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


def _pydantic_validate(model_cls: Type, data: dict):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)  # type: ignore[call-arg]
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(data)  # type: ignore[attr-defined]
    return model_cls(**data)


def load_live_settings(path: str | Path):
    from ..config_live import LiveSettings

    p = Path(path)
    text = os.path.expandvars(p.read_text(encoding="utf-8"))
    data = yaml.safe_load(text) or {}
    if hasattr(LiveSettings, "from_dict"):
        return LiveSettings.from_dict(data)
    return _pydantic_validate(LiveSettings, data)
