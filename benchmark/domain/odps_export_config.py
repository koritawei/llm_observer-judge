"""ODPS sample-export schedule config.

Domain layer. Contains:
  - :class:`OdpsExportJob` — one business export job
  - :class:`OdpsExportConfig` — overall schedule (many jobs, compatible with the old single sql)
  - :func:`fetch_odps_export_config` — merge RemoteConfig with the DB snapshot (RemoteConfig is not shielded)

Preferred RemoteConfig JSON (each job carries its own schedule; ``id`` may be omitted and defaults to ``default``)::

    {
      "jobs": [
        {
          "enabled": true,
          "schedule_time": "10:30",
          "schedule_timezone": "Asia/Shanghai",
          "limit": 200,
          "oss_prefix": "search-eval/benchmark_result/",
          "object_name": "benchmark_final_results.csv",
          "project_id": "cmpwln0qs04fo6j8eryfdgznb",
          "sql_template": "SELECT ... ds='{ds}' AND project_id='{project_id}' LIMIT {limit}"
        }
      ]
    }

Legacy compatibility: when ``jobs`` is absent, the top-level ``sql_template`` / ``limit`` /
``oss_prefix`` / ``project_id`` are treated as a single job (``id=default``). Top-level
``enabled`` / ``schedule_*`` still apply.

When top-level ``enabled`` is omitted, the schedule is on as soon as a runnable job
exists (enabled plus non-empty SQL).

``{ds}`` / ``{limit}`` inside ``sql_template`` are placeholders (literal replace). They
must appear as a pair, or be entirely absent so the SQL is self-contained.
``{project_id}`` is optional: when the template contains it, the value is injected from
``jobs[].project_id`` (or the top-level default) and filters ``$.record.project_id``.
"""

import re

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field, field_validator, model_validator

from benchmark.settings.remote_config_store import fetch_cached_settings
from benchmark.settings.setting import settings
from benchmark.domain._cache import get_or_compute
from benchmark.domain.runtime_snapshot import current_odps_export_config

logger = get_logger(__name__)

#: Default RemoteConfig coordinates for the ODPS export schedule (overridable from settings).
ODPS_EXPORT_DATA_ID = "agent-eval.odps.export"
ODPS_EXPORT_GROUP = "agent-eval.service.group"

_DEFAULT_SCHEDULE_TIME = "06:00"
#: Default export object name / schedule timezone, exported so runtime_config_defaults and
#: others can share them and the same literal does not drift (LINT-06).
DEFAULT_SCHEDULE_TIMEZONE = _DEFAULT_SCHEDULE_TIMEZONE = "Asia/Shanghai"
_DEFAULT_LIMIT = 200
_DEFAULT_JOB_ID = "default"
DEFAULT_OBJECT_NAME = _DEFAULT_OBJECT_NAME = "benchmark_final_results.csv"

#: Built-in default sample SQL (search_agent.stream.overview: JOIN first, then sample).
#: Placeholders ``{ds}`` / ``{limit}`` / ``{project_id}`` match the RemoteConfig example so
#: a ``--odps-ds`` / ``--oss-ds`` backfill injects the real partition. ``{project_id}`` comes
#: from RemoteConfig ``jobs[].project_id`` and filters ``$.record.project_id``; the CSV
#: ``project`` column is that JSON field. Do not switch back to a self-contained ``GETDATE()``
#: (it silently ignores the CLI date and is very hard to debug). See
#: ``docs/remote_config-odps-export-config-example.json``.
DEFAULT_SQL_TEMPLATE = """\
WITH overview_raw AS (
    SELECT  GET_JSON_OBJECT(content_json, '$.record.id') AS record_id
           ,GET_JSON_OBJECT(content_json, '$.record.output') AS raw_output
           ,GET_JSON_OBJECT(content_json, '$.record.project_id') AS project
    FROM    (
                SELECT  SPLIT_PART(content, CHR(1), 1) AS content_json
                FROM    autonavi_ai.s_tt_dwd_gd_ai_chat_req_langfuse_trace_raw_ri_tt4
                WHERE   ds = '{ds}'
            )
    WHERE   GET_JSON_OBJECT(content_json, '$.record.name') = 'search_agent.stream.overview'
      AND   GET_JSON_OBJECT(content_json, '$.record.id') IS NOT NULL
      AND   GET_JSON_OBJECT(content_json, '$.record.output') IS NOT NULL
      AND   GET_JSON_OBJECT(content_json, '$.record.output') <> ''
      AND   GET_JSON_OBJECT(content_json, '$.record.project_id') = '{project_id}'
)
,joined AS (
    SELECT  GET_JSON_OBJECT(t.root_input, '$.query') AS query
           ,t.langfuse_trace_id AS trace_id
           ,GET_JSON_OBJECT(t.root_input, '$.adiu') AS adiu
           ,GET_JSON_OBJECT(t.root_input, '$.longitude') AS longitude
           ,GET_JSON_OBJECT(t.root_input, '$.latitude') AS latitude
           ,GET_JSON_OBJECT(t.root_output, '$.address') AS address
           ,CONCAT(
                TO_CHAR(
                    FROM_UNIXTIME(CAST(t.trace_start_time / 1000 AS BIGINT))
                   ,'yyyy-mm-dd hh:mi:ss'
                )
               ,'.'
               ,LPAD(CAST(t.trace_start_time % 1000 AS STRING), 3, '0')
            ) AS current_time
           ,GET_JSON_OBJECT(t.root_output, '$.user_profile') AS user_profile
           ,GET_JSON_OBJECT(t.root_output, '$.app_use_context') AS app_use_context
           ,COALESCE(
                NULLIF(GET_JSON_OBJECT(t.root_output, '$.result'), '')
               ,NULLIF(s.raw_output, '')
            ) AS result
           ,t.tool_calls_json AS tool_call_json
           ,s.project AS project
    FROM    overview_raw s
    JOIN    autonavi_ai.dwd_gd_ai_chat_langfuse_trace_di t
    ON      t.ds = '{ds}'
     AND   t.langfuse_trace_id = s.record_id
    WHERE   GET_JSON_OBJECT(t.root_input, '$.query') IS NOT NULL
      AND   GET_JSON_OBJECT(t.root_input, '$.query') <> ''
)
SELECT  *
FROM    joined
ORDER BY RAND()
LIMIT   {limit}
"""

#: job.id: starts with a lowercase letter, only [a-z0-9._-], length 1..64.
_JOB_ID_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
#: Value injected for SQL ``{project_id}``: a Langfuse project id. No quotes, so a literal replace cannot inject.
SQL_PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def normalize_sql_project_id(value: object) -> str:
    """Normalize and validate the ``project_id`` injected into SQL. An empty string means unset."""
    text = str(value or "").strip()
    if not text:
        return ""
    if not SQL_PROJECT_ID_PATTERN.match(text):
        msg = (
            f"project_id 非法: {value!r}（须匹配 "
            r"^[A-Za-z0-9_-]{1,128}$，禁止引号/空白）"
        )
        raise ValueError(msg)
    return text


class OdpsExportJob(BaseModel):
    """One business ODPS → CSV → OSS export job.

    Attributes:
        id: Business id (lock key / CLI ``--job``). Omitted means ``default``.
        enabled: Whether it joins the schedule, default ``True``.
        schedule_time / schedule_timezone: This job's fixed time. Empty inherits the top level.
        limit: Sample count. ``None`` inherits top-level ``OdpsExportConfig.limit``.
        oss_prefix: OSS prefix. Empty inherits the top-level ``oss_prefix``.
        object_name: Object file name (basename only), default ``benchmark_final_results.csv``.
        project_id: Langfuse project. Empty inherits the top level. Injected into the SQL
            filter when the template contains ``{project_id}``.
        sql_template: Sample SQL. Empty means the scheduler skips this job.
    """

    model_config = {"extra": "ignore"}

    id: str = _DEFAULT_JOB_ID
    enabled: bool = True
    schedule_time: str = ""
    schedule_timezone: str = ""
    limit: int | None = None
    oss_prefix: str = ""
    object_name: str = _DEFAULT_OBJECT_NAME
    project_id: str = ""
    sql_template: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def _default_id(cls, value: object) -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            return _DEFAULT_JOB_ID
        return str(value)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        text = (value or "").strip()
        if not _JOB_ID_PATTERN.match(text):
            msg = (
                f"job.id 非法: {value!r}（须匹配 "
                r"^[a-z][a-z0-9._-]{0,63}$）"
            )
            raise ValueError(msg)
        return text

    @field_validator("object_name")
    @classmethod
    def _validate_object_name(cls, value: str) -> str:
        text = (value or "").strip() or _DEFAULT_OBJECT_NAME
        # Reject path traversal: keep only the last segment.
        name = text.replace("\\", "/").split("/")[-1].strip()
        if not name or name in {".", ".."}:
            return _DEFAULT_OBJECT_NAME
        return name

    @field_validator("project_id", mode="before")
    @classmethod
    def _normalize_project_id(cls, value: object) -> str:
        return normalize_sql_project_id(value)


class OdpsExportConfig(BaseModel):
    """ODPS sample-export schedule (many jobs, compatible with the old single SQL).

    Attributes:
        enabled: Whether in-process scheduled export is on. ``None`` (omitted) turns it
            on when a runnable job exists.
        schedule_time / schedule_timezone: Top-level default schedule, inherited when a job omits them.
        limit / oss_prefix / project_id / sql_template: Legacy single-business fields. With no
            ``jobs``, they synthesize the single ``id=default`` job. With ``jobs``, they are
            the default limit / prefix / project_id.
        jobs: Multi-business job list. When non-empty it takes precedence over the top-level ``sql_template``.
    """

    model_config = {"extra": "ignore"}

    enabled: bool | None = None
    schedule_time: str = _DEFAULT_SCHEDULE_TIME
    schedule_timezone: str = _DEFAULT_SCHEDULE_TIMEZONE
    limit: int = _DEFAULT_LIMIT
    oss_prefix: str = ""
    project_id: str = ""
    sql_template: str = ""
    jobs: list[OdpsExportJob] = Field(default_factory=list)

    @field_validator("project_id", mode="before")
    @classmethod
    def _normalize_project_id(cls, value: object) -> str:
        return normalize_sql_project_id(value)

    @model_validator(mode="after")
    def _reject_duplicate_job_ids(self) -> "OdpsExportConfig":
        seen: set[str] = set()
        for job in self.jobs:
            if job.id in seen:
                msg = f"jobs 中存在重复 id: {job.id!r}"
                raise ValueError(msg)
            seen.add(job.id)
        return self

    def scheduler_enabled(self) -> bool:
        """Whether the scheduler should start / trigger.

        - ``enabled is False``: forced off.
        - ``enabled is True``: on.
        - ``enabled is None`` (omitted in RemoteConfig): on when a runnable job exists.
        """
        if self.enabled is False:
            return False
        if self.enabled is True:
            return True
        if self.enabled_jobs():
            return True
        return bool(self.sql_template.strip())

    def resolved_jobs(self) -> list[OdpsExportJob]:
        """Expand the schedulable job list (inherited limit / oss_prefix / project_id / schedule filled in).

        - Non-empty ``jobs``: return them in array order (including ``enabled=False``; the caller filters).
        - Otherwise, if the top-level ``sql_template`` is non-empty: return the single ``default`` job.
        - Otherwise an empty list.
        """
        if self.jobs:
            return [self._materialize_job(job) for job in self.jobs]
        if self.sql_template.strip():
            return [
                self._materialize_job(
                    OdpsExportJob(
                        id=_DEFAULT_JOB_ID,
                        enabled=True,
                        limit=self.limit,
                        oss_prefix=self.oss_prefix,
                        object_name=_DEFAULT_OBJECT_NAME,
                        project_id=self.project_id,
                        sql_template=self.sql_template,
                    ),
                ),
            ]
        return []

    def enabled_jobs(self) -> list[OdpsExportJob]:
        """Only jobs with ``enabled=True`` and a non-empty ``sql_template``."""
        out: list[OdpsExportJob] = []
        for job in self.resolved_jobs():
            if not job.enabled:
                continue
            if not job.sql_template.strip():
                logger.warning(
                    "odps_export_config.skip_job id=%s (empty sql_template)",
                    job.id,
                )
                continue
            out.append(job)
        return out

    def get_job(self, job_id: str) -> OdpsExportJob | None:
        """Find a materialized job by id (including disabled)."""
        target = (job_id or "").strip()
        for job in self.resolved_jobs():
            if job.id == target:
                return job
        return None

    def _materialize_job(self, job: OdpsExportJob) -> OdpsExportJob:
        limit = job.limit if job.limit is not None else self.limit
        prefix = job.oss_prefix.strip() or self.oss_prefix
        project_id = (job.project_id or "").strip() or self.project_id
        schedule_time = (job.schedule_time or "").strip() or self.schedule_time
        schedule_timezone = (job.schedule_timezone or "").strip() or self.schedule_timezone
        return job.model_copy(
            update={
                "limit": limit if limit and limit > 0 else _DEFAULT_LIMIT,
                "oss_prefix": prefix,
                "object_name": job.object_name or _DEFAULT_OBJECT_NAME,
                "project_id": project_id,
                "schedule_time": schedule_time,
                "schedule_timezone": schedule_timezone,
            },
        )


def fetch_odps_export_config(
    *,
    data_id: str | None = None,
    group: str | None = None,
    use_cache: bool = True,
) -> OdpsExportConfig:
    """Merge RemoteConfig ``agent-eval.odps.export`` with the DB snapshot (ODPS is **not** shielded).

    - RemoteConfig is still the base source (behavior matches history when the page does not write the DB).
    - A DB ``kind=odps_job`` row written by the page or API overrides the RemoteConfig job
      with the same id. Jobs that exist only in RemoteConfig are kept.
    - When RemoteConfig is unavailable or unset, fall back to the DB snapshot (no jobs = the scheduler does not run).

    Args:
        data_id: RemoteConfig data_id, default read from settings.
        group: RemoteConfig group, default read from settings.
        use_cache: Whether to use the RemoteConfig parse cache (TTL 300s).
    """
    snapshot = current_odps_export_config()
    db_config = snapshot if isinstance(snapshot, OdpsExportConfig) else OdpsExportConfig()
    remote_config = _fetch_remote_config_odps_config(
        data_id=data_id,
        group=group,
        use_cache=use_cache,
    )
    if not remote_config.jobs and not remote_config.sql_template.strip():
        return db_config
    merged: dict[str, OdpsExportJob] = {job.id: job for job in remote_config.resolved_jobs()}
    for job in db_config.resolved_jobs():
        merged[job.id] = job
    return OdpsExportConfig(
        enabled=remote_config.enabled,
        schedule_time=remote_config.schedule_time,
        schedule_timezone=remote_config.schedule_timezone,
        limit=remote_config.limit,
        oss_prefix=remote_config.oss_prefix,
        project_id=remote_config.project_id,
        sql_template=remote_config.sql_template,
        jobs=list(merged.values()),
    )


def _fetch_remote_config_odps_config(
    *,
    data_id: str | None,
    group: str | None,
    use_cache: bool,
) -> OdpsExportConfig:
    """Load and parse the RemoteConfig ODPS config (300s parse cache; an empty config on failure)."""
    target_id = data_id or settings.odps_export_remote_config_data_id
    target_group = group or settings.odps_export_remote_config_group

    def _load_and_parse() -> OdpsExportConfig:
        raw = fetch_cached_settings(target_id, target_group, use_cache=use_cache)
        if not raw:
            return OdpsExportConfig()
        try:
            return OdpsExportConfig.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "RemoteConfig odps export config parse failed (%s/%s): %s",
                target_id,
                target_group,
                exc,
            )
            return OdpsExportConfig()

    return get_or_compute(
        (target_id, target_group),
        _load_and_parse,
        use_cache=use_cache,
    )


__all__ = [
    "DEFAULT_OBJECT_NAME",
    "DEFAULT_SQL_TEMPLATE",
    "DEFAULT_SCHEDULE_TIMEZONE",
    "ODPS_EXPORT_DATA_ID",
    "ODPS_EXPORT_GROUP",
    "OdpsExportConfig",
    "OdpsExportJob",
    "SQL_PROJECT_ID_PATTERN",
    "fetch_odps_export_config",
    "normalize_sql_project_id",
]
