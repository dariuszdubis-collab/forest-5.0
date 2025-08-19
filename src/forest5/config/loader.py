from __future__ import annotations

from pathlib import Path
import os
import yaml

from typing import Type


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
    return _pydantic_validate(LiveSettings, data)
