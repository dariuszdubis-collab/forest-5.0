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
    base_dir = p.parent

    ai_data = data.get("ai")
    if ai_data:
        context_file = ai_data.get("context_file")
        if context_file:
            context_path = Path(context_file)
            if not context_path.is_absolute():
                ai_data["context_file"] = str((base_dir / context_path).resolve())
        data["ai"] = ai_data

    time_data = data.get("time")
    if time_data:
        model_data = time_data.get("model")
        if model_data:
            model_path = model_data.get("path")
            if model_path:
                model_path_obj = Path(model_path)
                if not model_path_obj.is_absolute():
                    model_data["path"] = str((base_dir / model_path_obj).resolve())
            time_data["model"] = model_data
        data["time"] = time_data

    if hasattr(LiveSettings, "from_dict"):
        return LiveSettings.from_dict(data)
    return _pydantic_validate(LiveSettings, data)
