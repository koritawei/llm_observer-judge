import json
from pathlib import Path

import pytest
import yaml

from benchmark.kernel.confloader import ConfigError, load_conf
from benchmark.kernel.lease import ACQUIRE_LUA

CONF = Path(__file__).resolve().parents[1] / "conf"


def test_conf_rejects_secret_material(tmp_path: Path):
    target = tmp_path / "app.local.yaml"
    target.write_text("version: 1\npassword: hunter2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        yaml.safe_load(target.read_text())
        from benchmark.kernel.confloader import _load_yaml

        _load_yaml(target, {"version", "password"})


def test_conf_rejects_unknown_mode():
    conf = load_conf(CONF, "local")
    conf.bindings["bindings"]["execution"]["exact_match_judge"]["mode"] = "mystery"
    with pytest.raises(ConfigError):
        conf.binding("execution", "exact_match_judge")


def test_eval_domain_does_not_fall_back_to_isolation():
    conf = load_conf(CONF, "local")
    assert conf.eval_domain("local-eval") is None
    assert conf.domain("local-eval")["id"] == "local-eval"
    assert conf.isolation["replay_domain"] == "local-eval"


def test_lease_script_is_single_key():
    assert "ZREMRANGEBYSCORE" not in ACQUIRE_LUA
    assert "DEL" not in ACQUIRE_LUA
    assert "KEYS[1]" in ACQUIRE_LUA


def test_source_has_no_vendor_identifiers():
    banned = ("aone", "amap", "alibaba", "ieval", "diamond", "normandy", "agentgraph", "langfuse")
    roots = [
        CONF.parent / "benchmark" / "kernel",
        CONF.parent / "benchmark" / "defaults",
        CONF.parent / "front",
        CONF,
    ]
    hits = []
    for root in roots:
        for path in root.rglob("*"):
            if (
                "libs" in path.parts
                or "agents" in path.parts
                or "label_studio" in path.parts
                or path.name in {"app.daily.yaml", "app.pre.yaml", "app.prod.yaml"}
                or path.suffix not in {".py", ".yaml", ".html", ".js", ".css"}
            ):
                continue
            text = path.read_text(encoding="utf-8").lower()
            for word in banned:
                if word in text:
                    hits.append(f"{path}:{word}")
    assert hits == []
