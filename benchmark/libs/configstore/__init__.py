"""YAML stand-in for the remote config client. Reads ``conf/``."""

from pathlib import Path

import yaml


class RemoteConfigClient:
    def __init__(self, root: str | Path = "conf"):
        self.root = Path(root)

    def get_config(self, data_id: str, group: str = "DEFAULT_GROUP") -> dict:
        del group
        path = self.root / data_id
        if path.suffix == "":
            path = path.with_suffix(".yaml")
        if not path.is_file():
            return {}
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must be a mapping")
        return loaded
