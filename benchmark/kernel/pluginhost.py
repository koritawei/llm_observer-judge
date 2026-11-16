"""Freeze plugin refs at publish time and load them back by snapshot hash."""

from __future__ import annotations

import hashlib
import hmac
import importlib
from typing import Any

from benchmark.kernel.confloader import Conf, canonical, secret, sha256_text
from benchmark.kernel.errors import ConfigError

HOST_API_VERSION = "1"
REQUIRED_SLOTS = (
    "source",
    "normalizer",
    "admission_policy",
    "retry_policy",
    "capacity_policy",
    "trigger_policy",
    "attribution",
    "dispatch_policy",
    "fusion",
    "aggregation",
    "completion_extra",
    "report",
    "callback_authenticator",
)


class PluginHost:
    def __init__(self, conf: Conf):
        self.conf = conf
        self._trust = secret(conf.app["plugin_signature"]["secret_ref"])

    def freeze(self, kind: str, name: str) -> dict[str, Any]:
        resolved = self.conf.binding(kind, name)
        body = {
            "plugin_id": resolved["name"],
            "kind": kind,
            "version": resolved["version"],
            "class_ref": resolved.get("class_ref") or "",
            "artifact_hash": resolved["artifact_hash"],
            "config_hash": sha256_text(canonical(resolved.get("config") or {})),
            "config": resolved.get("config") or {},
            "mode": resolved["mode"],
            "signer": "hmac",
            "host_api_version": HOST_API_VERSION,
            "fallback_local": resolved.get("fallback_local", False),
        }
        if resolved.get("base_url"):
            body["base_url"] = resolved["base_url"]
            body["url_key"] = resolved.get("url_key")
        body["signature"] = self._sign(body["artifact_hash"])
        return body

    def load(self, frozen: dict[str, Any]) -> Any:
        self._verify(frozen)
        if frozen.get("mode") == "remote":
            from benchmark.kernel.remote import RemoteAdapter

            return RemoteAdapter(frozen)
        module_name, _, qual = str(frozen["class_ref"]).partition(":")
        if not module_name.startswith(("benchmark.", "business.")) and module_name not in {
            "benchmark",
            "business",
        }:
            raise ConfigError(f"refusing to import {frozen['class_ref']}")
        module = importlib.import_module(module_name)
        cls = getattr(module, qual)
        return cls(frozen.get("config") or {})

    def _sign(self, artifact_hash: str) -> str:
        return hmac.new(self._trust.encode("utf-8"), artifact_hash.encode("utf-8"), hashlib.sha256).hexdigest()

    def _verify(self, frozen: dict[str, Any]) -> None:
        if frozen.get("host_api_version") != HOST_API_VERSION:
            raise ConfigError("plugin host api mismatch")
        digest = hmac.new(
            self._trust.encode("utf-8"),
            str(frozen.get("artifact_hash", "")).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(digest, str(frozen.get("signature") or "")):
            raise ConfigError("plugin signature mismatch")
        if frozen.get("mode") != "remote" and frozen.get("class_ref"):
            from benchmark.kernel.confloader import _file_hash

            if _file_hash(frozen["class_ref"]) != frozen["artifact_hash"]:
                raise ConfigError("frozen artifact hash does not match the installed file")
