"""OdpsExportConfig multi-job parsing and compatibility unit tests."""

import json
from pathlib import Path

import pytest
from benchmark.domain.odps_export_config import OdpsExportConfig, OdpsExportJob
from pydantic import ValidationError


def test_legacy_sql_template_becomes_default_job() -> None:
    cfg = OdpsExportConfig.model_validate(
        {
            "enabled": True,
            "limit": 50,
            "oss_prefix": "p",
            "sql_template": "SELECT 1 WHERE ds='{ds}' LIMIT {limit}",
        },
    )
    jobs = cfg.enabled_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "default"
    assert jobs[0].limit == 50
    assert jobs[0].oss_prefix == "p"


def test_jobs_array_preferred_over_top_level_sql() -> None:
    cfg = OdpsExportConfig.model_validate(
        {
            "enabled": True,
            "sql_template": "SELECT legacy",
            "limit": 200,
            "oss_prefix": "top-prefix",
            "jobs": [
                {
                    "id": "biz-a",
                    "sql_template": "SELECT a WHERE ds='{ds}' LIMIT {limit}",
                    "limit": 10,
                    "oss_prefix": "prefix-a",
                },
                {
                    "id": "biz-b",
                    "enabled": False,
                    "sql_template": "SELECT b WHERE ds='{ds}' LIMIT {limit}",
                },
                {
                    "id": "biz-c",
                    "sql_template": "SELECT c WHERE ds='{ds}' LIMIT {limit}",
                    # inherit limit/prefix
                },
            ],
        },
    )
    enabled = cfg.enabled_jobs()
    assert [j.id for j in enabled] == ["biz-a", "biz-c"]
    assert enabled[0].limit == 10
    assert enabled[0].oss_prefix == "prefix-a"
    assert enabled[1].limit == 200
    assert enabled[1].oss_prefix == "top-prefix"


def test_duplicate_job_id_rejected() -> None:
    with pytest.raises(ValidationError, match="重复"):
        OdpsExportConfig.model_validate(
            {
                "jobs": [
                    {"id": "same", "sql_template": "SELECT 1"},
                    {"id": "same", "sql_template": "SELECT 2"},
                ],
            },
        )


def test_invalid_job_id_rejected() -> None:
    with pytest.raises(ValidationError):
        OdpsExportJob.model_validate({"id": "Bad_ID", "sql_template": "x"})


def test_object_name_strips_path() -> None:
    job = OdpsExportJob.model_validate(
        {"id": "x", "object_name": "../../evil.csv", "sql_template": "SELECT 1"},
    )
    assert job.object_name == "evil.csv"


def test_empty_sql_job_skipped_in_enabled_jobs() -> None:
    cfg = OdpsExportConfig.model_validate(
        {
            "jobs": [
                {"id": "ok", "sql_template": "SELECT 1 WHERE ds='{ds}' LIMIT {limit}"},
                {"id": "empty", "sql_template": "  "},
            ],
        },
    )
    assert [j.id for j in cfg.enabled_jobs()] == ["ok"]


def test_jobs_only_remote_config_format_without_id_or_top_enabled() -> None:
    """Matches live RemoteConfig: only jobs[], with no id and no top-level enabled."""
    cfg = OdpsExportConfig.model_validate(
        {
            "jobs": [
                {
                    "enabled": True,
                    "schedule_time": "10:30",
                    "schedule_timezone": "Asia/Shanghai",
                    "limit": 200,
                    "oss_prefix": "search-eval/benchmark_result/",
                    "sql_template": "SELECT 1 WHERE ds='{ds}' LIMIT {limit}",
                },
            ],
        },
    )
    assert cfg.scheduler_enabled() is True
    jobs = cfg.enabled_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "default"
    assert jobs[0].schedule_time == "10:30"
    assert jobs[0].oss_prefix == "search-eval/benchmark_result/"
    assert jobs[0].project_id == ""


def test_job_project_id_parsed_and_inherited() -> None:
    cfg = OdpsExportConfig.model_validate(
        {
            "project_id": "top_project",
            "jobs": [
                {
                    "id": "own",
                    "project_id": "cmpwln0qs04fo6j8eryfdgznb",
                    "sql_template": "SELECT '{project_id}' WHERE ds='{ds}' LIMIT {limit}",
                },
                {
                    "id": "inherit",
                    "sql_template": "SELECT '{project_id}' WHERE ds='{ds}' LIMIT {limit}",
                },
            ],
        },
    )
    jobs = {job.id: job for job in cfg.enabled_jobs()}
    assert jobs["own"].project_id == "cmpwln0qs04fo6j8eryfdgznb"
    assert jobs["inherit"].project_id == "top_project"


def test_invalid_project_id_rejected() -> None:
    with pytest.raises(ValidationError, match="project_id"):
        OdpsExportJob.model_validate(
            {"id": "x", "project_id": "'; DROP TABLE x; --", "sql_template": "SELECT 1"},
        )


def test_remote_config_example_json_parses_project_id() -> None:
    """The docs example JSON matches the live RemoteConfig shape, including project_id and SQL placeholders."""
    from benchmark.reporting.framework.odps_export import DEFAULT_SQL_TEMPLATE, render_sql

    path = Path(__file__).resolve().parents[3] / "docs" / "remote_config-odps-export-config-example.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = OdpsExportConfig.model_validate(raw)
    jobs = cfg.enabled_jobs()
    assert len(jobs) == 1
    assert jobs[0].project_id == "cmpwln0qs04fo6j8eryfdgznb"
    assert "{project_id}" in jobs[0].sql_template
    assert "GET_JSON_OBJECT(content_json, '$.record.project_id')" in jobs[0].sql_template
    assert (
        "GET_JSON_OBJECT(content_json, '$.record.project_id') = '{project_id}'"
        in jobs[0].sql_template
    )
    sql = render_sql(
        jobs[0].sql_template,
        "20260908",
        jobs[0].limit or 200,
        project_id=jobs[0].project_id,
    )
    assert (
        "GET_JSON_OBJECT(content_json, '$.record.project_id') = 'cmpwln0qs04fo6j8eryfdgznb'"
    ) in sql
    assert "{project_id}" not in sql
    assert jobs[0].sql_template.strip() == DEFAULT_SQL_TEMPLATE.strip()


def test_explicit_enabled_false_overrides_jobs() -> None:
    cfg = OdpsExportConfig.model_validate(
        {
            "enabled": False,
            "jobs": [
                {
                    "id": "x",
                    "sql_template": "SELECT 1 WHERE ds='{ds}' LIMIT {limit}",
                },
            ],
        },
    )
    assert cfg.scheduler_enabled() is False


def test_empty_config_scheduler_disabled() -> None:
    assert OdpsExportConfig().scheduler_enabled() is False
