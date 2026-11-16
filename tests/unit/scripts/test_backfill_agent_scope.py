"""Pure functions of backfill_agent_scope: missing-scope detection, kind labels, and building backfill candidates."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "backfill_agent_scope.py"


def _load_module():  # noqa: ANN202
    name = "backfill_agent_scope_under_test"
    stub = types.ModuleType("_bootstrap")
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"_bootstrap": stub}):
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


backfill = _load_module()


def test_missing_agent_scope_detects_empty_and_blank() -> None:
    assert backfill.missing_agent_scope(SimpleNamespace(agent_scope=None))
    assert backfill.missing_agent_scope(SimpleNamespace(agent_scope=""))
    assert backfill.missing_agent_scope(SimpleNamespace(agent_scope="   "))
    assert not backfill.missing_agent_scope(SimpleNamespace(agent_scope="amap-agent"))


def test_registration_kind_plain_vs_customized() -> None:
    assert backfill.registration_kind(SimpleNamespace(customization=None)) == "plain"
    assert backfill.registration_kind(SimpleNamespace(customization={"judge": {}})) == "customized"


def test_prepare_skips_when_scope_present() -> None:
    assert (
        backfill.prepare_scope_backfill(
            {"id": "ba-1", "name": "example", "agentScope": "amap-agent"},
            default_scope="amap-agent",
            validate=lambda payload: SimpleNamespace(
                id=payload["id"],
                name=payload["name"],
                customization=None,
                model_dump=lambda **_: payload,
            ),
        )
        is None
    )


def test_prepare_validated_path_for_plain_missing_scope() -> None:
    dumped = {"id": "ba-1", "name": "example", "agentScope": "amap-agent"}

    class _Doc:
        id = "ba-1"
        name = "example"
        customization = None

        def model_dump(self, **_: object) -> dict[str, object]:
            return dumped

    candidate = backfill.prepare_scope_backfill(
        {"id": "ba-1", "name": "example"},
        default_scope="amap-agent",
        validate=lambda _payload: _Doc(),
    )
    assert candidate is not None
    assert candidate.via == "validated"
    assert candidate.kind == "plain"
    assert candidate.payload["agentScope"] == "amap-agent"


def test_prepare_raw_patch_when_customized_still_invalid() -> None:
    """Dirty pre-env data: if validation still fails after injecting a scope, degrade to raw_patch and do not skip."""

    def _boom(_payload: dict[str, object]) -> object:
        raise ValueError("启用 prompt/skill 定制必须填写 agentScope")

    payload = {
        "id": "example_benchmark--example_benchmark--default--v1",
        "name": "example_benchmark",
        "customization": {"judge": {"stage": "judge"}},
        "generateAttributionReport": False,
    }
    candidate = backfill.prepare_scope_backfill(
        payload,
        default_scope="amap-agent",
        validate=_boom,
    )
    assert candidate is not None
    assert candidate.via == "raw_patch"
    assert candidate.kind == "customized"
    assert candidate.document_id == "example_benchmark--example_benchmark--default--v1"
    assert candidate.payload["agentScope"] == "amap-agent"
    assert candidate.payload["customization"] == payload["customization"]
