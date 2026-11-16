"""Cluster-report constants — three-bucket map, batch-tag regex, and default thresholds.

The three-bucket map stays aligned with ``generate-eval-report.py`` (scale drift is guarded
by the T13 comparison test). The script hardcodes ``MODEL_BUCKET_MAP`` / ``MODEL_BUCKETS``;
the service keeps a copy. If either side changes and the other does not follow, the
comparison test fails.
"""

import re

#: Batch-tag shape: run-<8-digit date>-<6-digit time>, with an optional custom suffix (for example -whitebox-v3).
#: Same source as the evaluation-side schema.BATCH_TAG_RE.
BATCH_TAG_RE = re.compile(r"^run-(\d{8})-(\d{6})(?:-.*)?$")

#: Three-bucket order (aligned with script-side MODEL_BUCKETS).
MODEL_BUCKETS: tuple[str, ...] = ("cgi", "react", "verify")

#: Bucket → display label (used as xlsx column names, aligned with script-side BUCKET_LABELS).
BUCKET_LABELS: dict[str, str] = {
    "cgi": "意图",
    "react": "React",
    "verify": "Verify",
}

#: observation name → bucket. Aligned entry by entry with ``_OBS_NAME_TO_BUCKET`` in generate-eval-report.py
#: (including both 2.0 and 3.0 naming). This service aggregates only 3.0 composite Judge data; the 2.0 map is kept
#: so the two sides of the comparison test (T13) stay identical.
OBSERVATION_NAME_TO_BUCKET: dict[str, str] = {
    # 2.0
    "model_call.unifiedIntentAgent": "cgi",
    "model_call.toolAgentReact": "react",
    "model_call.toolAgent": "react",
    "react.tool_select": "react",
    "model_call.verifyAgent": "verify",
    # 3.0
    "cgi_classify": "cgi",
    "AmapChatModel.chat": "react",
}

#: D0-D6 dimension order (aligned with evaluation-side models._DIMENSIONS).
DIMENSIONS: tuple[str, ...] = ("D0", "D1", "D2", "D3", "D4", "D5", "D6")


#: Dimension display names shared across HTML, Excel, and stats enhancements.
DIMENSION_LABELS: dict[str, str] = {
    "D0": "安全与合规",
    "D1": "意图识别",
    "D2": "需求约束满足度",
    "D3": "信息真实性与可信度",
    "D4": "任务落地与执行闭环",
    "D5": "表达与场景适配",
    "D6": "信息充分性与决策帮助度",
}

#: Status set that counts as a "dimension failure"; effectiveness and crosstab share this scale.
FAILURE_STATUSES: frozenset[str] = frozenset({"fail", "partial", "ambiguous"})

#: The three "overturned evaluation" types in the closed attribution-conclusion set (a subset of badcase_attribution.schemas.CONCLUSION_TYPES).
OVERTURN_CONCLUSION_TYPES: frozenset[str] = frozenset(
    {
        "judge_false_positive",
        "eval_rule_issue",
        "dataset_issue",
    }
)

#: Machine-readable contract version of the report.
#: v2 = three score fields (judge / attribution / effective; primary effectiveness stats switch to ``effective_score``)
#:    + direct reads of the unified-envelope shared core (``business_id`` / ``scenario`` / ``profile_id`` /
#:      ``profile_version``, ``metric_observations``, ``diagnostic_findings``,
#:      and the L1 ``observations`` slice statistics).
REPORT_SCHEMA_VERSION = "cluster_report.v2"

#: Default width of the PG query time window (seconds): batch_tag start time + this window.
DEFAULT_QUERY_WINDOW_S = 6 * 3600
#: Lookback on the left of the time window (seconds): tolerate a small clock skew between the load generator and the DB.
QUERY_WINDOW_BACKOFF_S = 600

#: Default watcher parameters (seconds).
DEFAULT_JUDGE_TIMEOUT_S = 1800  # judge close timeout: 30min
DEFAULT_ATTRIBUTION_STALL_S = 300  # attribution stall: pending count unchanged for 5min
#: The report is an async job, so convergence does not need second-level sensitivity. 60s vs a 30min timeout loses only ~3% freshness,
#: but halves how often the watcher polls PG (count_progress uses a narrow-column path; see collector).
DEFAULT_POLL_INTERVAL_S = 60

#: Concurrency cap for Langfuse perf fetches. The Langfuse service has limited CPU/DB headroom; a high-concurrency burst
#: stalls it. The report is an async job and is not sensitive, so lower concurrency spreads the burst (T7).
DEFAULT_PERF_CONCURRENCY = 3
#: Minimum gap between per-trace fetch starts (seconds): extra trickle throttling on top of the concurrency cap,
#: so a freed concurrency slot does not immediately fire the next pair (each trace = get_trace + list_observations,
#: two requests). A 115-trace batch spreads its startup over at least ~57s (T7).
DEFAULT_PERF_MIN_INTERVAL_S = 0.5

#: Evaluator aggregated by default for P0 (composite Judge). The TraceRecord composite business key includes evaluator_name,
#: so one trace may have several rows. Every count must be narrowed by this, or judged_count is inflated.
DEFAULT_EVALUATOR_NAME = "vehicle_agent"

#: Three-tier thresholds of the D0-D6 total-score scale (aligned with `sv>=2.0 / sv>=1.0` in eval-compare.py).
#: Centralized here so stats modules do not each hardcode their own.
PASS_SCORE = 2.0
PARTIAL_SCORE = 1.0

#: Ungrouped xlsx sheet name (where rows land when case_meta.suite is missing).
UNGROUPED_SHEET = "未分组"

#: L2 clustering prompt (same pattern as the attribution side: conf/agents/<name>/prompts/<prompt>.prompt).
DEFAULT_CLUSTER_PROMPT_NAME: str = "cluster_summary"
CLUSTER_PROMPT_CONF_DIR: tuple[str, ...] = (
    "conf",
    "agents",
    "cluster_report_agent",
    "prompts",
)
PROMPT_FILE_SUFFIX: str = ".prompt"


__all__ = [
    "BATCH_TAG_RE",
    "BUCKET_LABELS",
    "CLUSTER_PROMPT_CONF_DIR",
    "DEFAULT_ATTRIBUTION_STALL_S",
    "DEFAULT_CLUSTER_PROMPT_NAME",
    "DEFAULT_EVALUATOR_NAME",
    "DEFAULT_JUDGE_TIMEOUT_S",
    "DEFAULT_PERF_CONCURRENCY",
    "DEFAULT_PERF_MIN_INTERVAL_S",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_QUERY_WINDOW_S",
    "DIMENSIONS",
    "DIMENSION_LABELS",
    "FAILURE_STATUSES",
    "MODEL_BUCKETS",
    "OBSERVATION_NAME_TO_BUCKET",
    "OVERTURN_CONCLUSION_TYPES",
    "PARTIAL_SCORE",
    "PASS_SCORE",
    "PROMPT_FILE_SUFFIX",
    "QUERY_WINDOW_BACKOFF_S",
    "REPORT_SCHEMA_VERSION",
    "UNGROUPED_SHEET",
]
