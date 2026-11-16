"""Live-LLM regression for Vehicle scenario tag classification.

Skipped by default; the daily evaluation model is called only when RUN_LIVE_LLM_TESTS=1 is set explicitly.
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from benchmark.settings import Settings
ClassifyTagNode = None
RENDER_CONTEXT = None
from benchmark.service.main import create_app
from tests.integration.helpers import async_lifespan

_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT = _SERVICE_ROOT.parents[1]
_PROMPTS_DIR = "conf/agents/judge_vehicle_agent/prompts"
_DATASET_DIR = _REPOSITORY_ROOT / "tmp" / "classify_tag"
_DATASET_PATH = _DATASET_DIR / "tag_query.csv"
_LIVE_LLM_ENV = "RUN_LIVE_LLM_TESTS"
_SAMPLE_SIZE_ENV = "LIVE_TAG_CLASSIFY_SAMPLE_SIZE"
_QUERY_FILTER_ENV = "LIVE_TAG_CLASSIFY_QUERIES"


@dataclass(frozen=True)
class _Case:
    expected_tag: str
    query: str
    env: str


def _load_cases() -> list[_Case]:
    env_by_trip_state = {
        "行前": (_DATASET_DIR / "行前_env.txt").read_text(encoding="utf-8"),
        "行中": (_DATASET_DIR / "行中_env.txt").read_text(encoding="utf-8"),
    }
    with _DATASET_PATH.open(encoding="utf-8", newline="") as dataset:
        rows = csv.DictReader((line for line in dataset if line.strip()), delimiter="\t")
        return [
            _Case(
                expected_tag=row["intent"].strip(),
                query=row["Query"].strip(),
                env=env_by_trip_state[row["行前or行中"].strip()],
            )
            for row in rows
            if row["intent"].strip() and row["Query"].strip()
        ]


def _filter_cases(cases: list[_Case], queries: str) -> list[_Case]:
    selected_queries = {query.strip() for query in queries.split(",") if query.strip()}
    if not selected_queries:
        return cases
    return [case for case in cases if case.query in selected_queries]


def _sample_cases(cases: list[_Case], sample_size: int) -> list[_Case]:
    """Sample by rotating tags, preferring broader scenario coverage over the first few CSV rows."""
    if sample_size <= 0 or sample_size >= len(cases):
        return cases

    grouped: dict[str, list[_Case]] = {}
    for case in cases:
        grouped.setdefault(case.expected_tag, []).append(case)

    sampled: list[_Case] = []
    index = 0
    while len(sampled) < sample_size:
        added = False
        for tag_cases in grouped.values():
            if index < len(tag_cases):
                sampled.append(tag_cases[index])
                added = True
                if len(sampled) == sample_size:
                    return sampled
        if not added:
            return sampled
        index += 1
    return sampled


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_llm_classifies_tag_query_dataset(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no preset tag, the live model's classifications must match the human-labeled baseline set item by item."""
    if os.environ.get(_LIVE_LLM_ENV) != "1":
        pytest.skip(f"set {_LIVE_LLM_ENV}=1 to call the real evaluator LLM")

    monkeypatch.setenv("AGENT_EVAL_MQ_ENABLED", "false")
    app = create_app(settings=Settings(database_url=f"sqlite:///{tmp_path}/eval.db"))
    sample_size = int(os.environ.get(_SAMPLE_SIZE_ENV, "0"))
    cases = _filter_cases(_load_cases(), os.environ.get(_QUERY_FILTER_ENV, ""))
    cases = _sample_cases(cases, sample_size)

    mismatches: list[str] = []
    try:
        async with async_lifespan(app):
            evaluator = app.state.runtime.agent_manager.get("judge_vehicle_agent")
            node = ClassifyTagNode(evaluator._llm, _PROMPTS_DIR)

            for case in cases:
                patch = await node({RENDER_CONTEXT: {"query": case.query, "env": case.env}})
                actual_tag = patch.get("tag", "")
                if actual_tag != case.expected_tag:
                    mismatches.append(
                        f"expected={case.expected_tag}, actual={actual_tag}, query={case.query}"
                    )
    except AttributeError as exc:
        if "daily_cluster_report_scheduler" not in str(exc):
            raise

    assert not mismatches, (
        f"tag 分类准确率={(len(cases) - len(mismatches)) / len(cases):.1%} "
        f"({len(cases) - len(mismatches)}/{len(cases)})\n" + "\n".join(mismatches)
    )
