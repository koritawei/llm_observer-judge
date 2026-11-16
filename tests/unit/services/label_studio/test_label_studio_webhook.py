"""Unit tests for WebhookService — annotation parsing + dual-bucket write-back."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from benchmark.labeling.webhook_service import (
    LabelStudioWebhookService,
    _extract_annotation_fields,
)


def _make_webhook_service() -> LabelStudioWebhookService:
    resolver = MagicMock()
    resolver.is_known_project.return_value = True
    trace_svc = AsyncMock()
    trace_svc.update_ls_annotation.return_value = SimpleNamespace(trace_id="t1")
    return LabelStudioWebhookService(
        project_resolver=resolver,
        trace_record_service=trace_svc,
    )


class TestExtractAnnotationFields:
    def test_all_fields(self) -> None:
        result = [
            {"from_name": "judge_score_correct", "value": {"choices": ["no"]}},
            {"from_name": "corrected_score", "value": {"text": ["0.8"]}},
            {"from_name": "corrected_reason", "value": {"text": ["wrong"]}},
            {"from_name": "triage_level_correct", "value": {"choices": ["yes"]}},
            {"from_name": "corrected_triage_level", "value": {"choices": ["mild_bad"]}},
            {"from_name": "attribution_correct", "value": {"choices": ["partial"]}},
            {"from_name": "corrected_root_cause", "value": {"text": ["tool_error"]}},
            {"from_name": "corrected_owner_group", "value": {"text": ["Toolbox"]}},
        ]
        fields = _extract_annotation_fields(result)
        assert fields["judge_score_correct"] == "no"
        assert fields["corrected_score"] == "0.8"
        assert fields["corrected_reason"] == "wrong"
        assert fields["triage_level_correct"] == "yes"
        assert fields["corrected_triage_level"] == "mild_bad"
        assert fields["machine_attribution_correct"] == "partial"
        assert fields["primary_cause"] == "tool_error"
        assert fields["owner_group"] == "Toolbox"

    def test_missing_fields_return_none(self) -> None:
        fields = _extract_annotation_fields([])
        assert fields["judge_score_correct"] is None
        assert fields["primary_cause"] is None

    def test_empty_choices(self) -> None:
        result = [{"from_name": "judge_score_correct", "value": {"choices": []}}]
        fields = _extract_annotation_fields(result)
        assert fields["judge_score_correct"] is None


class TestHandleAnnotation:
    async def test_success(self) -> None:
        svc = _make_webhook_service()
        annotation = {
            "result": [
                {"from_name": "judge_score_correct", "value": {"choices": ["yes"]}},
            ],
            "completed_by": 7,
            "created_at": "2026-07-07T10:00:00Z",
        }
        task = {"data": {"trace_record_id": "uuid-123", "trace_id": "t1"}}
        result = await svc.handle_annotation(annotation, {"id": 1}, task)
        assert result["status"] == "ok"
        svc._trace_record_service.update_ls_annotation.assert_called_once()
        call_kw = svc._trace_record_service.update_ls_annotation.call_args.kwargs
        assert call_kw["record_id"] == "uuid-123"
        judge = call_kw["judge_annotation_data"]
        assert judge["triage_result"] == {"raw": annotation["result"]}
        assert judge["triage_annotator"] == "7"
        attr = call_kw["attribution_annotation_data"]
        assert attr["attribution_result"] == {"raw": annotation["result"]}
        assert attr["annotator"] == "7"
        assert attr["annotated_at"] == "2026-07-07T10:00:00Z"

    async def test_missing_project_returns_422(self) -> None:
        svc = _make_webhook_service()
        annotation = {"result": []}
        task = {"data": {"trace_id": "t1"}}
        result = await svc.handle_annotation(annotation, None, task)
        assert result["status"] == "invalid"
        assert result["_http_status"] == 422

    async def test_missing_keys_returns_422(self) -> None:
        svc = _make_webhook_service()
        annotation = {"result": []}
        task = {"data": {"trace_id": "t1"}}
        result = await svc.handle_annotation(annotation, {"id": 1}, task)
        assert result["_http_status"] == 422

    async def test_unknown_project_ignored(self) -> None:
        svc = _make_webhook_service()
        svc._project_resolver.is_known_project.return_value = False
        annotation = {"result": []}
        task = {"data": {}}
        result = await svc.handle_annotation(annotation, {"id": 999}, task)
        assert result["status"] == "ignored"

    async def test_record_not_found_returns_404(self) -> None:
        svc = _make_webhook_service()
        svc._trace_record_service.update_ls_annotation.return_value = None
        annotation = {"result": []}
        task = {"data": {"trace_record_id": "uuid-404", "trace_id": "t1"}}
        result = await svc.handle_annotation(annotation, {"id": 1}, task)
        assert result["_http_status"] == 404
