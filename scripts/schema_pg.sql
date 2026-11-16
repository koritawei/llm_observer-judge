-- trace_pipeline_records CREATE TABLE statement (PostgreSQL)
-- Source: services/agent-eval/app/data_model/models/trace_record.py
-- Usage: psql -h <host> -U <user> -d <db> -f scripts/schema_pg.sql

CREATE TABLE IF NOT EXISTS trace_pipeline_records (
    id              UUID            NOT NULL PRIMARY KEY,
    created_at      TIMESTAMPTZ     NOT NULL,
    updated_at      TIMESTAMPTZ     NOT NULL,

    -- Identity (composite unique key)
    trace_id        VARCHAR(128)    NOT NULL,
    project_id      VARCHAR(128),
    evaluator_name  VARCHAR(512)    NOT NULL,
    agent_name      VARCHAR(128),

    -- Pipeline status (values constrained by the TraceStatus enum)
    status          VARCHAR(10)     NOT NULL,

    -- Data-source scene (aligned with ORM TraceRecord.scene: voice=voice, non_voice=non-voice)
    scene           VARCHAR(64)     NOT NULL DEFAULT 'non_voice',

    -- Report-stat scalars (query hot path)
    owner_group     VARCHAR(128),
    triage_level    VARCHAR(32),     -- values constrained by the TriageLevel enum
    root_cause      VARCHAR(256),
    notification_job_status VARCHAR(32),     -- synced by the observer from the notification_job_json bucket

    -- 7 JSON buckets
    judge_json                                       JSONB,
    attribution_detail_json                          JSONB,
    notification_dispatch_json                               JSONB,
    notification_job_json                                    JSONB,
    label_studio_attribution_annotation_json        JSONB,
    label_studio_badcase_judge_annotation_json       JSONB,
    trace_meta_json                                  JSONB,

    -- Composite unique key
    CONSTRAINT uq_trace_pipeline_records_record
        UNIQUE (trace_id, project_id, agent_name, evaluator_name)
);

-- Indexes (query hot path)
CREATE INDEX IF NOT EXISTS idx_trace_records_status
    ON trace_pipeline_records (status);
CREATE INDEX IF NOT EXISTS idx_trace_records_triage_level
    ON trace_pipeline_records (triage_level);
CREATE INDEX IF NOT EXISTS idx_trace_records_root_cause
    ON trace_pipeline_records (root_cause);
CREATE INDEX IF NOT EXISTS idx_trace_records_notification_job_status
    ON trace_pipeline_records (notification_job_status);
CREATE INDEX IF NOT EXISTS idx_trace_records_owner_group
    ON trace_pipeline_records (owner_group);
CREATE INDEX IF NOT EXISTS idx_trace_records_created_at
    ON trace_pipeline_records (created_at);
