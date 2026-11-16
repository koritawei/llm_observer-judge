"""Load conf/**/*.yaml. Runtime publish freezes the result; workers do not reread it."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from benchmark.kernel.errors import ConfigError

_SECRET_KEYS = re.compile(
    r"(password|secret|token|bearer|access_key|api_key|private_key)$",
    re.IGNORECASE,
)
_CLASS_REF = re.compile(r"^(benchmark|business)(\.[a-z0-9_]+)*:[A-Za-z0-9_]+$")

APP_KEYS = {
    "version",
    "env",
    "keyspace_generation",
    "queue_prefix",
    "worker_profile",
    "allow_auto_artifact_hash",
    "redis_url",
    "database",
    "plugin_signature",
    "object_store_dir",
    "slots",
    "retry",
    "slot_busy",
    "healer",
    "limits",
    "llm_max_concurrency",
    "instance_max_concurrency",
    "run_max_duration_ms",
    "discover_on_catalog_error",
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_secrets(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key != "secret_ref" and _SECRET_KEYS.search(str(key)):
                raise ConfigError(f"secret material is not allowed in yaml: {path}.{key}")
            _reject_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}[{index}]")
    elif isinstance(value, str) and value.lower().startswith("bearer "):
        raise ConfigError(f"secret material is not allowed in yaml: {path}")


def _load_yaml(path: Path, allowed: set[str] | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    if allowed is not None:
        unknown = set(data) - allowed
        if unknown:
            raise ConfigError(f"unknown fields in {path.name}: {sorted(unknown)}")
    _reject_secrets(data, path.name)
    return data


def secret(ref: str) -> str:
    env = f"EVALBENCH_SECRET_{ref.upper()}"
    value = os.environ.get(env, "")
    if not value:
        raise ConfigError(f"missing secret env {env}")
    return value


@dataclass
class Conf:
    root: Path
    env_name: str
    app: dict[str, Any]
    urls: dict[str, Any]
    isolation: dict[str, Any]
    eval_domains: dict[str, Any]
    l2_business: list[str]
    bindings: dict[str, Any]
    enabled: set[str]
    disabled: set[str]
    aliases_batch: dict[str, Any]
    aliases_streaming: dict[str, Any]
    profiles: dict[str, Any]
    groups: dict[str, Any]
    file_hashes: dict[str, str] = field(default_factory=dict)

    def database_dsn(self) -> str:
        db = self.app["database"]
        password = secret(db["secret_ref"])
        return (
            f"postgresql://{db['user']}:{password}@{db['host']}:{db['port']}/{db['name']}"
        )

    def redis_url(self) -> str:
        return str(self.app["redis_url"])

    def queue_prefix(self) -> str:
        return str(self.app["queue_prefix"])

    def domain(self, domain_id: str) -> dict[str, Any] | None:
        for item in self.isolation.get("domains", []):
            if item.get("id") == domain_id:
                return item
        return None

    def eval_domain(self, domain_id: str) -> dict[str, Any] | None:
        return (self.eval_domains.get("eval_domains") or {}).get(domain_id)

    def dataset_domain(self, name: str) -> dict[str, Any] | None:
        found = (self.eval_domains.get("domains") or {}).get(name)
        return found if isinstance(found, dict) else None

    def domain_datasets(self, name: str) -> list[dict[str, Any]]:
        spec = self.dataset_domain(name) or {}
        rows = []
        for dataset_id in spec.get("datasets") or []:
            item = self.eval_domain(str(dataset_id))
            if item:
                rows.append({"id": str(dataset_id), **item})
        return rows

    def url_entry(self, key: str) -> dict[str, Any]:
        entry = (self.urls.get("urls") or {}).get(key)
        if not isinstance(entry, dict) or not (entry.get("base_url") or entry.get("endpoint")):
            raise ConfigError(f"unknown url key: {key}")
        return entry

    def class_ref(self, alias: str, *, streaming: bool = False) -> str:
        table = self.aliases_streaming if streaming else self.aliases_batch
        found = (table.get("aliases") or {}).get(alias)
        if not found or not found.get("class_ref"):
            raise ConfigError(f"alias is not on the allowlist: {alias}")
        ref = str(found["class_ref"])
        if not _CLASS_REF.match(ref):
            raise ConfigError(f"class_ref is outside the framework packages: {ref}")
        return ref

    def binding(self, kind: str, name: str) -> dict[str, Any]:
        if name in self.disabled or name not in self.enabled:
            raise ConfigError(f"plugin is not enabled: {name}")
        raw = ((self.bindings.get("bindings") or {}).get(kind) or {}).get(name)
        if not isinstance(raw, dict):
            raise ConfigError(f"missing binding {kind}.{name}")
        mode = _normalize_mode(str(raw.get("mode", "")))
        if raw.get("enabled") is False:
            default_name = raw.get("default")
            if not default_name:
                raise ConfigError(f"disabled binding {kind}.{name} has no default")
            return self.binding(kind, str(default_name))
        alias = str(raw.get("alias") or name)
        streaming = kind == "stream_filter"
        resolved: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "mode": mode,
            "fallback_local": bool(raw.get("fallback_local", False)),
            "config": raw.get("config") or {},
            "version": str(raw.get("version") or "1"),
        }
        if mode == "remote":
            url_key = raw.get("url_key")
            if not url_key:
                raise ConfigError(f"remote binding {name} requires url_key")
            entry = self.url_entry(str(url_key))
            resolved["url_key"] = url_key
            resolved["base_url"] = entry.get("base_url") or entry.get("endpoint")
            resolved["class_ref"] = ""
            declared = str(raw.get("artifact_hash") or "")
            if declared in ("", "auto"):
                raise ConfigError(f"remote binding {name} requires an artifact hash")
            resolved["artifact_hash"] = declared
        else:
            resolved["class_ref"] = self.class_ref(alias, streaming=streaming)
            digest = _file_hash(resolved["class_ref"])
            declared = str(raw.get("artifact_hash") or "auto")
            if declared not in ("auto", digest):
                raise ConfigError(f"artifact hash mismatch for {name}")
            if declared == "auto" and not self.app.get("allow_auto_artifact_hash"):
                raise ConfigError(f"explicit artifact hash required for {name}")
            resolved["artifact_hash"] = digest
        return resolved

    def profile(self, profile_id: str) -> dict[str, Any]:
        found = (self.profiles.get("profiles") or {}).get(profile_id)
        if not isinstance(found, dict):
            raise ConfigError(f"unknown profile: {profile_id}")
        return found


def _normalize_mode(value: str) -> str:
    aliases = {
        "platform": "local",
        "inprocess": "local",
        "local": "local",
        "external_http": "remote",
        "http": "remote",
        "rpc": "remote",
        "remote": "remote",
    }
    if value not in aliases:
        raise ConfigError(f"unknown plugin mode: {value}")
    return aliases[value]


def _file_hash(class_ref: str) -> str:
    module_name = class_ref.split(":", 1)[0]
    module = __import__(module_name, fromlist=["_"])
    path = getattr(module, "__file__", None)
    if not path:
        raise ConfigError(f"cannot hash {class_ref}")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return "sha256:" + digest


def load_conf(root: Path | str, env_name: str | None = None) -> Conf:
    root = Path(root)
    env_name = env_name or os.environ.get("EVALBENCH_ENV", "local")
    kernel_app = root / "kernel" / f"app.{env_name}.yaml"
    app_path = kernel_app if kernel_app.is_file() else root / f"app.{env_name}.yaml"
    app = _load_yaml(app_path, APP_KEYS)
    urls = _load_yaml(root / f"urls.{env_name}.yaml", {"version", "urls"})
    isolation = _load_yaml(
        root / "isolation_domains.yaml",
        {"version", "domains", "replay_domain", "allow_request_override_aliases"},
    )
    eval_domains = _load_yaml(root / "eval_domains.yaml", {"version", "eval_domains", "domains"})
    l2 = _load_yaml(root / "l2_business.yaml", {"version", "l2_business"})
    bindings = _load_yaml(root / "plugins" / "bindings.yaml", {"version", "bindings"})
    enabled_doc = _load_yaml(root / "plugins" / "enabled.yaml", {"version", "enabled", "disabled"})
    aliases_batch = _load_yaml(root / "plugins" / "aliases.batch.yaml", {"version", "aliases"})
    aliases_streaming = _load_yaml(
        root / "plugins" / "aliases.streaming.yaml", {"version", "aliases"}
    )
    profiles = _load_yaml(root / "profiles.yaml", {"version", "profiles"})
    groups = _load_yaml(root / "plugins" / "groups.yaml", {"version", "businesses"})
    _validate_urls(urls)
    _validate_groups(groups)
    _validate_domains(isolation, urls)
    replay = isolation.get("replay_domain")
    if replay and not any(item.get("id") == replay for item in isolation.get("domains", [])):
        raise ConfigError("replay_domain is not declared in isolation_domains")
    conf = Conf(
        root=root,
        env_name=env_name,
        app=app,
        urls=urls,
        isolation=isolation,
        eval_domains=eval_domains,
        l2_business=list(l2.get("l2_business") or []),
        bindings=bindings,
        enabled=set(enabled_doc.get("enabled") or []),
        disabled=set(enabled_doc.get("disabled") or []),
        aliases_batch=aliases_batch,
        aliases_streaming=aliases_streaming,
        profiles=profiles,
        groups=groups,
    )
    conf.file_hashes = {
        "app": sha256_text(canonical(app)),
        "urls": sha256_text(canonical(urls)),
        "isolation": sha256_text(canonical(isolation)),
        "eval_domains": sha256_text(canonical(eval_domains)),
        "l2": sha256_text(canonical(l2)),
        "plugins": sha256_text(canonical({"bindings": bindings, "enabled": enabled_doc})),
        "profiles": sha256_text(canonical(profiles)),
        "groups": sha256_text(canonical(groups)),
    }
    return conf


def _validate_groups(groups: dict[str, Any]) -> None:
    businesses = groups.get("businesses") or {}
    if not isinstance(businesses, dict):
        raise ConfigError("businesses must be a mapping")
    for name, spec in businesses.items():
        listed = spec.get("groups") if isinstance(spec, dict) else None
        if not isinstance(listed, list) or not listed:
            raise ConfigError(f"business {name} requires groups")
        if not all(isinstance(item, str) and item for item in listed):
            raise ConfigError(f"business {name} has an empty group")


def _validate_urls(urls: dict[str, Any]) -> None:
    for key, entry in (urls.get("urls") or {}).items():
        if not isinstance(entry, dict):
            raise ConfigError(f"url {key} must be a mapping")
        raw = entry.get("base_url") or entry.get("endpoint")
        if not raw:
            raise ConfigError(f"url {key} needs base_url or endpoint")
        parsed = urlparse(str(raw))
        if parsed.scheme not in {"http", "https", "file"} or not (parsed.netloc or parsed.scheme == "file"):
            raise ConfigError(f"url {key} is not an absolute http(s) or file url")


def _validate_domains(isolation: dict[str, Any], urls: dict[str, Any]) -> None:
    known = set((urls.get("urls") or {}))
    for domain in isolation.get("domains") or []:
        if not domain.get("id") or not domain.get("header_name") or not domain.get("queue_prefix"):
            raise ConfigError("isolation domain requires id, header_name, queue_prefix")
        for key in domain.get("allowed_url_keys") or []:
            if key not in known:
                raise ConfigError(f"domain {domain['id']} references unknown url {key}")
