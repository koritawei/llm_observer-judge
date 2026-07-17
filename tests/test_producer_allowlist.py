"""R-02: producer allowlist enforcement."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from llmobserverjudge.contracts.topics import (
    Component,
    ProducerForbiddenError,
    Topic,
    assert_can_publish,
)
from llmobserverjudge.shared.mq import InMemoryMQ


@pytest.mark.asyncio
async def test_eval_worker_cannot_publish_eval_jobs() -> None:
    mq = InMemoryMQ()
    with pytest.raises(ProducerForbiddenError):
        await mq.publish(Component.EVAL_WORKER, Topic.EVAL_JOBS, {"x": 1})


@pytest.mark.asyncio
async def test_ingest_cannot_publish_eval_jobs() -> None:
    mq = InMemoryMQ()
    with pytest.raises(ProducerForbiddenError):
        await mq.publish(Component.INGEST, Topic.EVAL_JOBS, {"x": 1})


@pytest.mark.asyncio
async def test_dispatch_can_publish_eval_jobs() -> None:
    mq = InMemoryMQ()
    await mq.publish(Component.DISPATCH, Topic.EVAL_JOBS, {"ok": True})
    assert await mq.depth(Topic.EVAL_JOBS) == 1


@pytest.mark.asyncio
async def test_gateway_publishes_spans_only() -> None:
    mq = InMemoryMQ()
    await mq.publish(Component.GATEWAY, Topic.SPANS, {"span_id": "a"})
    with pytest.raises(ProducerForbiddenError):
        await mq.publish(Component.GATEWAY, Topic.EVAL_JOBS, {})


def test_eval_worker_source_has_no_eval_jobs_publish() -> None:
    """Static guard: eval_worker must not call mq.publish with EVAL_JOBS."""
    root = Path(__file__).resolve().parents[1] / "src" / "llmobserverjudge" / "eval_worker"
    for path in root.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_publish = (
                isinstance(func, ast.Attribute) and func.attr == "publish"
            ) or (isinstance(func, ast.Name) and func.id == "publish")
            if not is_publish:
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if (
                    isinstance(arg, ast.Attribute)
                    and arg.attr == "EVAL_JOBS"
                ) or (
                    isinstance(arg, ast.Attribute)
                    and isinstance(arg.value, ast.Name)
                    and arg.value.id == "Topic"
                    and arg.attr == "EVAL_JOBS"
                ):
                    pytest.fail(f"forbidden publish(EVAL_JOBS) call in {path}")


def test_assert_can_publish_helper() -> None:
    assert_can_publish(Component.DISPATCH, Topic.ANNOTATION_JOBS)
    with pytest.raises(ProducerForbiddenError):
        assert_can_publish(Component.EVAL_WORKER, Topic.ANNOTATION_JOBS)
