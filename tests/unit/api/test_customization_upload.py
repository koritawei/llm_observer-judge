"""Customization upload endpoint orchestration: multipart assembly → upload service → camelCase ref."""

from io import BytesIO
from typing import Any

import pytest
from benchmark.routes import benchmark_console as benchmark_console_api
from benchmark.routes.benchmark_console import (
    CustomizationUploadForm,
    preview_customization,
    upload_customization,
)
from benchmark.domain import customization_config as customization_config_module
from benchmark.domain.customization_config import CustomizationConfig
from benchmark.customization import service as service_module
from benchmark.utils.error_handler import install_error_handlers
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient


class _FakeGitStore:
    def __init__(self, config: CustomizationConfig) -> None:
        self.commits: list[dict[str, Any]] = []

    async def commit_bundle(self, *, git_path: str, files: dict[str, bytes], message: str) -> str:
        self.commits.append({"git_path": git_path, "files": files, "message": message})
        return "deadbeefcafe"


@pytest.fixture()
def _enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    config = CustomizationConfig(enabled=True)
    monkeypatch.setattr(customization_config_module, "fetch_customization_config", lambda: config)
    monkeypatch.setattr(service_module, "fetch_customization_config", lambda: config)
    monkeypatch.setattr(service_module, "GitContentStore", _FakeGitStore)


def _form(**kwargs: object) -> CustomizationUploadForm:
    """Assemble upload text fields (aliases match the live Form parameters; test field names are simplified)."""
    values: dict[str, object] = {
        "agentName": kwargs.pop("agent_name", "example_benchmark"),
        "business": kwargs.pop("business", "biz"),
        "variantKey": kwargs.pop("variant_key", "v1"),
        "stage": kwargs.pop("stage", "judge"),
    }
    by_name = {
        "prompt_text": "promptText",
        "slot": "slot",
        "created_by": "createdBy",
    }
    for name, alias in by_name.items():
        if name in kwargs:
            values[alias] = kwargs[name]
    return CustomizationUploadForm(**values)  # type: ignore[arg-type]


async def test_upload_prompt_text_returns_camel_ref(_enabled: None) -> None:
    body = await upload_customization(
        _form(prompt_text="业务评测模板", slot="prompt", created_by="tester"),
    )
    assert body["contentVersion"] == 1
    assert body["gitCommit"] == "deadbeefcafe"
    assert body["gitPath"].endswith("/judge/v1")
    assert body["prompt"]["slots"] == ["prompt"]
    assert body["createdBy"] == "tester"


async def test_upload_prompt_file_uses_filename_as_slot(_enabled: None) -> None:
    upload = UploadFile(file=BytesIO("文件模板".encode()), filename="prompt.md")
    body = await upload_customization(
        _form(),
        prompt_files=[upload],
    )
    assert body["prompt"]["slots"] == ["prompt"]


async def test_upload_skill_files_strip_skill_prefix(_enabled: None) -> None:
    skill = UploadFile(file=BytesIO(b"# rule"), filename="skill/SKILL.md")
    body = await upload_customization(
        _form(stage="attribution"),
        skill_files=[skill],
    )
    assert body["skill"]["entry"] == "SKILL.md"


async def test_upload_policy_deny_maps_to_403(_enabled: None) -> None:
    with pytest.raises(HTTPException) as exc:
        await upload_customization(
            _form(
                agent_name="unknown_business_agent",
                prompt_text="模板",
                slot="prompt",
            ),
        )
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "customization_not_allowed"


async def test_upload_prompt_text_requires_slot(_enabled: None) -> None:
    with pytest.raises(HTTPException) as exc:
        await upload_customization(_form(prompt_text="模板"))
    assert exc.value.status_code == 400
    assert exc.value.detail["code"] == "slot_required"


async def test_preview_reads_single_file(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeStore:
        async def read_files(self, *, git_commit: str, paths: list[str]) -> dict[str, bytes]:
            return {paths[0]: "预览内容".encode()}

    monkeypatch.setattr(
        customization_config_module,
        "fetch_customization_config",
        lambda: CustomizationConfig(enabled=True),
    )
    monkeypatch.setattr(
        benchmark_console_api,
        "fetch_customization_config",
        lambda: CustomizationConfig(enabled=True),
    )
    monkeypatch.setattr(
        benchmark_console_api,
        "GitContentStore",
        lambda config: _FakeStore(),
    )
    body = await preview_customization(
        git_path="p/v1", git_commit="abc", file_path="prompt/prompts/prompt.md"
    )
    assert body["text"] == "预览内容"
    assert body["path"] == "prompt/prompts/prompt.md"


def _http_client() -> TestClient:
    """Assemble the real router and error handlers, and go through HTTP multipart rather than calling the function directly."""
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(benchmark_console_api.router)
    return TestClient(app)


def test_multipart_fields_reach_endpoint(_enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP layer: multipart text fields are parsed one by one (regression for mixing a Form model with File)."""
    captured: list[Any] = []

    async def _fake_upload(request: Any) -> dict[str, Any]:
        captured.append(request)
        return {
            "stage": request.stage,
            "content_version": 1,
            "git_branch": "master",
            "git_path": "customizations/example_benchmark/biz/v1/judge/v1",
            "git_commit": "deadbeefcafe",
            "bundle_sha256": "sha",
            "prompt": {"slots": sorted(request.prompt_texts)},
            "skill": None,
            "created_at": "2026-10-23T00:00:00Z",
            "created_by": request.created_by,
        }

    monkeypatch.setattr(service_module, "upload_stage_customization", _fake_upload)
    response = _http_client().post(
        "/v1/runtime-config/console/benchmark-agents/customizations",
        data={
            "agentName": "example_benchmark",
            "business": "biz",
            "variantKey": "v1",
            "stage": "judge",
            "promptText": "业务评测模板",
            "slot": "prompt",
            "createdBy": "tester",
        },
        files={"promptFiles": ("core.md", "文件模板".encode())},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["createdBy"] == "tester"
    assert body["stage"] == "judge"
    # Both text fields and the file reach the service: the slot comes from promptText, and the filename is the other slot.
    assert captured[0].agent_name == "example_benchmark"
    assert captured[0].variant_key == "v1"
    assert captured[0].prompt_texts == {"prompt": "业务评测模板", "core": "文件模板"}


def test_multipart_missing_field_reports_real_name(_enabled: None) -> None:
    """When a field is missing, the error's param points at the real field name, not the wrapper model name ``form``."""
    response = _http_client().post(
        "/v1/runtime-config/console/benchmark-agents/customizations",
        data={"business": "biz", "variantKey": "v1", "stage": "judge"},
        files={"promptFiles": ("prompt.md", b"x")},
    )
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == "agentName"
    assert error["type"] == "missing_required_parameter"
