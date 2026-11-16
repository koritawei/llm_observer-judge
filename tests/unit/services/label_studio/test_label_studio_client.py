"""Unit tests for LabelStudioClient — HTTP mocks only."""

from unittest.mock import patch

import httpx
import pytest
from benchmark.labeling.client import (
    LabelStudioAuth,
    LabelStudioClient,
    _RegisterWebhookSpec,
)

_DUMMY_REQ = httpx.Request("POST", "http://ls:8080/")


def _resp(status: int, *, json: dict | list) -> httpx.Response:
    return httpx.Response(status, json=json, request=_DUMMY_REQ)


@pytest.fixture()
def client() -> LabelStudioClient:
    return LabelStudioClient(
        base_url="http://ls:8080",
        auth=LabelStudioAuth(api_key="test-key"),
    )


class TestCreateProject:
    def test_success(self, client: LabelStudioClient) -> None:
        resp = _resp(200, json={"id": 42, "title": "t"})
        with patch.object(client._http, "request", return_value=resp):
            result = client.create_project("t", "<View/>")
        assert result["id"] == 42

    def test_error_raises(self, client: LabelStudioClient) -> None:
        resp = _resp(400, json={"detail": "bad"})
        with patch.object(client._http, "request", return_value=resp):
            with pytest.raises(httpx.HTTPStatusError):
                client.create_project("t", "<View/>")


class TestImportTasks:
    def test_success(self, client: LabelStudioClient) -> None:
        resp = _resp(200, json={"task_ids": [1, 2]})
        with patch.object(client._http, "request", return_value=resp):
            result = client.import_tasks(1, [{"data": {"x": 1}}])
        assert result["task_ids"] == [1, 2]


class TestRegisterWebhook:
    def test_send_payload_true(self, client: LabelStudioClient) -> None:
        resp = _resp(200, json={"id": 99})
        with patch.object(client._http, "request", return_value=resp) as mock_req:
            client.register_webhook(
                _RegisterWebhookSpec(
                    project_id=1,
                    url="http://cb/hook",
                    actions=["ANNOTATION_CREATED"],
                ),
            )
        body = mock_req.call_args.kwargs["json"]
        assert body["send_payload"] is True
        assert body["actions"] == ["ANNOTATION_CREATED"]

    def test_send_payload_explicit_false(self, client: LabelStudioClient) -> None:
        resp = _resp(200, json={"id": 99})
        with patch.object(client._http, "request", return_value=resp) as mock_req:
            client.register_webhook(
                _RegisterWebhookSpec(
                    project_id=1,
                    url="http://cb/hook",
                    actions=["ANNOTATION_CREATED"],
                    send_payload=False,
                ),
            )
        assert mock_req.call_args.kwargs["json"]["send_payload"] is False


class TestListProjects:
    def test_paginated_response(self, client: LabelStudioClient) -> None:
        resp = _resp(200, json={"results": [{"id": 1}, {"id": 2}]})
        with patch.object(client._http, "request", return_value=resp):
            result = client.list_projects()
        assert len(result) == 2

    def test_flat_list_response(self, client: LabelStudioClient) -> None:
        resp = _resp(200, json=[{"id": 1}])
        with patch.object(client._http, "request", return_value=resp):
            result = client.list_projects()
        assert len(result) == 1
