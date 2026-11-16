"""push entry profile-guard regression (locks the code-review fixes).

Locks three fix contracts:
- P0: the push default judge is always sync (local evaluation) — even if the process has an Aone factory
  assembled, Resolver's default inference must not flip it to async (push default semantics differ from pull);
- P1-2: push config_json always carries ``queue_pair="benchmark"`` (the basis for callback slot-release routing);
- P1-3: per_eval + async (judge/attribution) fails fast with 400 (a combination whose counts do not converge).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from benchmark.routes.benchmark import BenchmarkRunRequest, _prepare_push_run, _PushRunPrepare
from benchmark.runs.common.external_client_factory import (
    ExternalClientFactory,
    ExternalClientFactoryConfig,
)
from benchmark.runs.common.profile_resolver import ProfileResolver
from benchmark.runs.schema import JudgeProfileConfig
from benchmark.reporting.framework.base import BenchmarkComponentMetadata
from benchmark.reporting.framework.schema import AggregationConfig, Criterion
from fastapi import HTTPException

_META = BenchmarkComponentMetadata(
    name="demo_bench",
    service_name="demo",
    rubric=[Criterion(id="Q1", name="q", description="", weight=1.0, evaluator_ref="demo")],
    aggregation=AggregationConfig(method="weighted_avg", pass_threshold=0.5),
)


def _request_with_factory() -> MagicMock:
    """Request context with an Aone factory assembled (production shape)."""
    factory = ExternalClientFactory(
        ExternalClientFactoryConfig(
            judge_base_url="https://aone.example.com",
            api_key="ak-test",
            default_judge_pipeline_id="100",
        ),
    )
    runner = SimpleNamespace(profile_resolver=ProfileResolver(client_factory=factory))
    request = MagicMock()
    request.app.state.external_benchmark_runner = runner
    return request


def test_push_default_judge_stays_sync_even_with_factory() -> None:
    """P0 regression: the push default judge stays sync (an Aone factory being present does not flip it to async)."""
    prepared = _prepare_push_run(
        _request_with_factory(),
        _PushRunPrepare(
            benchmark_name="demo_bench",
            meta=_META,
            payload=BenchmarkRunRequest(),
            extra_config={},
        ),
    )
    assert prepared.resolved.judge.execution == "sync"
    assert prepared.resolved.judge.provider == "local"
    # Default attribution = none (live push has no per-item attribution)
    assert prepared.resolved.attribution.execution == "none"
    # Count model = benchmark progress columns (not external)
    assert prepared.external_count_model is False
    # The snapshot writes a sync judge (the basis for worker routing)
    assert prepared.config_json["judge"]["execution"] == "sync"


def test_push_config_json_always_marks_benchmark_queue_pair() -> None:
    """P1-2 regression: push config_json always carries queue_pair=benchmark."""
    prepared = _prepare_push_run(
        _request_with_factory(),
        _PushRunPrepare(
            benchmark_name="demo_bench",
            meta=_META,
            payload=BenchmarkRunRequest(),
            extra_config={},
        ),
    )
    assert prepared.config_json["queue_pair"] == "benchmark"


def test_push_config_json_carries_rule_link_url_from_rubric_class() -> None:
    """Scoring-rule link: an explicit rubric_ref → the Rubric class rule_link_url is written into config."""
    prepared = _prepare_push_run(
        _request_with_factory(),
        _PushRunPrepare(
            benchmark_name="demo_bench",
            meta=_META,
            payload=BenchmarkRunRequest(
                rubric_ref="business.runs.ieval.amap_chat.rubric:AmapChatRubricV2",
            ),
            extra_config={},
        ),
    )
    assert prepared.config_json["rule_link_url"] == (
        "https://alidocs.dingtalk.com/i/nodes/14dA3GK8gwAG5B1mHnbDXZPG89ekBD76"
    )


def test_push_config_json_rule_link_url_none_without_rubric_class() -> None:
    """No rubric_ref (meta.rubric fallback) → rule_link_url=None (the business did not configure one)."""
    prepared = _prepare_push_run(
        _request_with_factory(),
        _PushRunPrepare(
            benchmark_name="demo_bench",
            meta=_META,
            payload=BenchmarkRunRequest(),
            extra_config={},
        ),
    )
    assert prepared.config_json["rule_link_url"] is None


def test_push_explicit_async_judge_still_allowed_when_fused() -> None:
    """An explicit judge.target (same-host URL) → fused + async is allowed (the new capability is unaffected)."""
    prepared = _prepare_push_run(
        _request_with_factory(),
        _PushRunPrepare(
            benchmark_name="demo_bench",
            meta=_META,
            payload=BenchmarkRunRequest(
                judge=JudgeProfileConfig(
                    target="https://aone.example.com/api/pipelines/777/trigger",
                ),
            ),
            extra_config={},
        ),
    )
    assert prepared.resolved.judge.execution == "async"
    assert prepared.external_count_model is True


def test_push_per_eval_plus_async_rejected_400() -> None:
    """P1-3 regression: per_eval + async judge → 400 (a count-model combination that does not converge)."""
    from benchmark.runs.schema import TriageMode

    with pytest.raises(HTTPException) as exc:
        _prepare_push_run(
            _request_with_factory(),
            _PushRunPrepare(
                benchmark_name="demo_bench",
                meta=_META,
                payload=BenchmarkRunRequest(
                    triage_mode=TriageMode.PER_EVAL,
                    judge=JudgeProfileConfig(
                        target="https://aone.example.com/api/pipelines/777/trigger",
                    ),
                ),
                extra_config={},
            ),
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["reason"] == "invalid_mode_combination"
