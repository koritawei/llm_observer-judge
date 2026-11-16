from pathlib import Path
from typing import Any

import yaml


class AppConfigManager:
    _section_extensions: dict[tuple[str, str], Any] = {}

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.extensions: dict[str, Any] = {}
        self.sections: dict[str, Any] = {}

    def register_section(self, name: str, extension: Any = None, multi: bool = False, **_kwargs: Any) -> None:
        self.sections[name] = {"schema": extension, "multi": multi}
        if extension is not None:
            self.extensions[name] = extension

    def register_section_extension(self, name: str, extension: Any = None, schema: Any = None) -> None:
        if schema is None:
            self.extensions[name] = extension
            return
        self._section_extensions[(name, extension)] = schema

    def load_yaml(self, path: str) -> "AppConfigManager":
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            self.config.update(loaded)
        return self

    def section(self, name: str) -> Any:
        if name in self.config:
            return self.config[name]
        return self.extensions.get(name)

    def get_section(self, name: str, schema: Any = None) -> Any:
        raw = self.config.get(name) or {}
        if schema is None or not isinstance(raw, dict):
            return raw
        parsed = {}
        for key, value in raw.items():
            if isinstance(value, dict) and hasattr(schema, "model_validate"):
                parsed[key] = schema.model_validate(value)
            else:
                parsed[key] = value
        return parsed

    def get_section_extension(self, section: str, name: str, schema: Any) -> Any:
        raw = ((self.config.get(section) or {}).get("extensions") or {}).get(name) or {}
        if hasattr(schema, "model_validate"):
            return schema.model_validate(raw)
        return schema(**raw) if isinstance(raw, dict) else raw

    def get_resource(self, section: str, key: str, schema: Any) -> Any:
        raw = (self.config.get(section) or {}).get(key) or {}
        if hasattr(schema, "model_validate"):
            return schema.model_validate(raw if isinstance(raw, dict) else {})
        if isinstance(raw, dict):
            return schema(**raw)
        return raw

    def __getattr__(self, name: str) -> Any:
        config = self.__dict__.get("config") or {}
        if name in config:
            return config[name]
        raise AttributeError(name)


def register_section_extension(manager: AppConfigManager, name: str, extension: Any) -> None:
    manager.register_section_extension(name, extension)
