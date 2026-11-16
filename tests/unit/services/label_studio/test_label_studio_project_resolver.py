"""Unit tests for ProjectResolver — cache hit/miss, a Lock against stampede, and warm."""

import hashlib
from datetime import datetime
from unittest.mock import MagicMock

from benchmark.labeling.project_resolver import (
    _MAX_LS_TITLE_LEN,
    _TITLE_HASH_LEN,
    ProjectIdentity,
    ProjectResolver,
    ProjectResolverConfig,
)


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def _make_resolver(projects: list[dict] | None = None) -> ProjectResolver:
    client = MagicMock()
    client.list_projects.return_value = projects or []
    client.create_project.return_value = {"id": 99}
    client.register_webhook.return_value = {"id": 1}
    return ProjectResolver(
        client=client,
        config=ProjectResolverConfig(
            callback_url="http://localhost:8000",
            webhook_secret="secret",
        ),
    )


class TestResolve:
    async def test_cache_miss_creates_project(self) -> None:
        resolver = _make_resolver(projects=[])
        pid = await resolver.resolve(
            ProjectIdentity(
                agent_name="agent-a",
                project_id="proj-1",
            ),
        )
        assert pid == 99
        resolver._client.create_project.assert_called_once()

    async def test_cache_hit_no_api_call(self) -> None:
        resolver = _make_resolver(projects=[])
        await resolver.resolve(
            ProjectIdentity(
                agent_name="agent-a",
                project_id="proj-1",
            ),
        )
        resolver._client.create_project.reset_mock()
        pid = await resolver.resolve(
            ProjectIdentity(
                agent_name="agent-a",
                project_id="proj-1",
            ),
        )
        assert pid == 99
        resolver._client.create_project.assert_not_called()

    async def test_existing_project_found(self) -> None:
        title = f"{_today()}-proj-2-agent-b"
        resolver = _make_resolver(
            projects=[{"id": 42, "title": title}],
        )
        pid = await resolver.resolve(
            ProjectIdentity(
                agent_name="agent-b",
                project_id="proj-2",
            ),
        )
        assert pid == 42
        resolver._client.create_project.assert_not_called()


class TestDatasetNaming:
    async def test_dataset_name_and_run_name_used_in_title(self) -> None:
        resolver = _make_resolver(projects=[])
        pid = await resolver.resolve(
            ProjectIdentity(
                agent_name="agent-a",
                project_id="proj-1",
                dataset_name="ds-vehicle",
                run_name="run-001",
            ),
        )
        assert pid == 99
        title = resolver._client.create_project.call_args[0][0]
        assert title == f"{_today()}-ds-vehicle-run-001"

    async def test_dataset_naming_cache_hit(self) -> None:
        resolver = _make_resolver(projects=[])
        await resolver.resolve(
            ProjectIdentity(
                agent_name="agent-a",
                project_id="proj-1",
                dataset_name="ds-vehicle",
                run_name="run-001",
            ),
        )
        resolver._client.create_project.reset_mock()
        await resolver.resolve(
            ProjectIdentity(
                agent_name="agent-a",
                project_id="proj-1",
                dataset_name="ds-vehicle",
                run_name="run-001",
            ),
        )
        resolver._client.create_project.assert_not_called()

    async def test_partial_dataset_fields_fallback_to_default(self) -> None:
        resolver = _make_resolver(projects=[])
        # Only dataset_name, no run_name → should use default naming
        pid = await resolver.resolve(
            ProjectIdentity(
                agent_name="agent-a",
                project_id="proj-1",
                dataset_name="ds-vehicle",
                run_name=None,
            ),
        )
        assert pid == 99
        title = resolver._client.create_project.call_args[0][0]
        assert title == f"{_today()}-proj-1-agent-a"

    async def test_long_dataset_title_fits_ls_limit(self) -> None:
        resolver = _make_resolver(projects=[])
        dataset_name = "INJ-BLOCK-079-very-long-dataset-name"
        run_name = "nightly-regression-run-with-long-suffix"
        raw = f"{_today()}-{dataset_name}-{run_name}"
        assert len(raw) > _MAX_LS_TITLE_LEN

        await resolver.resolve(
            ProjectIdentity(
                agent_name="agent-a",
                project_id="proj-1",
                dataset_name=dataset_name,
                run_name=run_name,
            ),
        )
        title = resolver._client.create_project.call_args[0][0]
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:_TITLE_HASH_LEN]
        keep = _MAX_LS_TITLE_LEN - _TITLE_HASH_LEN - 1
        assert title == f"{raw[:keep]}-{digest}"
        assert len(title) == _MAX_LS_TITLE_LEN

    async def test_long_titles_with_different_suffixes_do_not_collide(self) -> None:
        resolver = _make_resolver(projects=[])
        prefix = "x" * 60
        t1 = resolver._build_title(
            ProjectIdentity(
                agent_name="a",
                project_id="p",
                dataset_name=prefix,
                run_name="run-a",
            ),
        )
        t2 = resolver._build_title(
            ProjectIdentity(
                agent_name="a",
                project_id="p",
                dataset_name=prefix,
                run_name="run-b",
            ),
        )
        assert len(t1) == _MAX_LS_TITLE_LEN
        assert len(t2) == _MAX_LS_TITLE_LEN
        assert t1 != t2


class TestWarm:
    async def test_warm_populates_cache(self) -> None:
        resolver = _make_resolver(
            projects=[
                {"id": 10, "title": f"{_today()}-proj-x-agent-x"},
                {"id": 20, "title": f"{_today()}-proj-y-agent-y"},
                {"id": 30, "title": "other-project"},
            ],
        )
        count = await resolver.warm()
        assert count == 2
        assert resolver._cache[f"{_today()}-proj-x-agent-x"] == 10
        assert resolver._cache[f"{_today()}-proj-y-agent-y"] == 20


class TestIsKnownProject:
    async def test_known_after_resolve(self) -> None:
        resolver = _make_resolver(projects=[])
        await resolver.resolve(
            ProjectIdentity(
                agent_name="agent-c",
                project_id="proj-3",
            ),
        )
        assert resolver.is_known_project(99) is True
        assert resolver.is_known_project(999) is False


class TestLabelConfigRouting:
    async def test_default_config_used(self) -> None:
        resolver = _make_resolver(projects=[])
        resolver._default_label_config = "conf/label_studio/badcase_review.xml"
        pid = await resolver.resolve(
            ProjectIdentity(
                agent_name="unknown-agent",
                project_id="proj-4",
            ),
        )
        assert pid == 99
        call_args = resolver._client.create_project.call_args
        label_config = (
            call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("label_config")
        )
        assert "judge_score_correct" in label_config

    async def test_custom_config_used(self, tmp_path) -> None:
        custom_xml = tmp_path / "custom.xml"
        custom_xml.write_text("<View><Text name='custom'/></View>")
        resolver = _make_resolver(projects=[])
        resolver._label_config_map = {"special-agent": str(custom_xml)}
        await resolver.resolve(
            ProjectIdentity(
                agent_name="special-agent",
                project_id="proj-5",
            ),
        )
        call_args = resolver._client.create_project.call_args
        label_config = (
            call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("label_config")
        )
        assert "custom" in label_config
