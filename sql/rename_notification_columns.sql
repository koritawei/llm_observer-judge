-- Rename dispatch columns and JSON keys to notification_*.
-- A database that already has the new column names skips the rename.

CREATE OR REPLACE FUNCTION rename_notification_json_keys(doc jsonb)
RETURNS jsonb
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    result jsonb := '{}'::jsonb;
    item_key text;
    item_value jsonb;
    next_key text;
    element jsonb;
    elements jsonb;
BEGIN
    IF doc IS NULL OR jsonb_typeof(doc) = 'null' THEN
        RETURN doc;
    END IF;
    IF jsonb_typeof(doc) = 'array' THEN
        elements := '[]'::jsonb;
        FOR element IN SELECT value FROM jsonb_array_elements(doc)
        LOOP
            elements := elements || jsonb_build_array(rename_notification_json_keys(element));
        END LOOP;
        RETURN elements;
    END IF;
    IF jsonb_typeof(doc) <> 'object' THEN
        RETURN doc;
    END IF;
    FOR item_key, item_value IN SELECT key, value FROM jsonb_each(doc)
    LOOP
        next_key := CASE item_key
            WHEN 'aone_ref' THEN 'notification_ref'
            WHEN 'aone_url' THEN 'notification_url'
            WHEN 'aone_assigned_to' THEN 'notification_assigned_to'
            WHEN 'aone_title' THEN 'notification_title'
            WHEN 'aone_dispatched_at' THEN 'notification_dispatched_at'
            WHEN 'aone_skip_reason' THEN 'notification_skip_reason'
            WHEN 'aone_job_status' THEN 'notification_job_status'
            WHEN 'aone_job_id' THEN 'notification_job_id'
            WHEN 'aone_job_idempotency_key' THEN 'notification_job_idempotency_key'
            WHEN 'aone_job_attempt_count' THEN 'notification_job_attempt_count'
            WHEN 'aone_job_next_attempt_at' THEN 'notification_job_next_attempt_at'
            WHEN 'aone_job_payload' THEN 'notification_job_payload'
            WHEN 'aone_job_result' THEN 'notification_job_result'
            WHEN 'aone_job_last_error' THEN 'notification_job_last_error'
            WHEN 'aone_project_id' THEN 'notification_project_id'
            WHEN 'aone_assignee' THEN 'notification_assignee'
            WHEN 'aone_participants' THEN 'notification_participants'
            WHEN 'aone_dispatch_enabled' THEN 'notification_dispatch_enabled'
            WHEN 'aoneDispatchEnabled' THEN 'notificationDispatchEnabled'
            WHEN 'aoneProjectId' THEN 'notificationProjectId'
            WHEN 'aoneAssignee' THEN 'notificationAssignee'
            WHEN 'aoneParticipants' THEN 'notificationParticipants'
            WHEN 'aone_page_url' THEN 'report_site_url'
            WHEN 'agent_aone_disabled' THEN 'agent_notification_disabled'
            ELSE item_key
        END;
        result := result || jsonb_build_object(next_key, rename_notification_json_keys(item_value));
    END LOOP;
    RETURN result;
END;
$$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'trace_pipeline_records' AND column_name = 'aone_job_status'
    ) THEN
        ALTER TABLE trace_pipeline_records
            RENAME COLUMN aone_job_status TO notification_job_status;
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'trace_pipeline_records' AND column_name = 'aone_dispatch_json'
    ) THEN
        ALTER TABLE trace_pipeline_records
            RENAME COLUMN aone_dispatch_json TO notification_dispatch_json;
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'trace_pipeline_records' AND column_name = 'aone_job_json'
    ) THEN
        ALTER TABLE trace_pipeline_records
            RENAME COLUMN aone_job_json TO notification_job_json;
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'benchmark_run_items' AND column_name = 'aone_dispatch_json'
    ) THEN
        ALTER TABLE benchmark_run_items
            RENAME COLUMN aone_dispatch_json TO notification_dispatch_json;
    END IF;
END $$;

ALTER INDEX IF EXISTS idx_trace_pipeline_records_aone_job_status
    RENAME TO idx_trace_pipeline_records_notification_job_status;
ALTER INDEX IF EXISTS idx_trace_records_aone_job_status
    RENAME TO idx_trace_records_notification_job_status;

UPDATE trace_pipeline_records
SET notification_dispatch_json = rename_notification_json_keys(notification_dispatch_json)
WHERE notification_dispatch_json IS NOT NULL;

UPDATE trace_pipeline_records
SET notification_job_json = rename_notification_json_keys(notification_job_json)
WHERE notification_job_json IS NOT NULL;

UPDATE benchmark_run_items
SET notification_dispatch_json = rename_notification_json_keys(notification_dispatch_json)
WHERE notification_dispatch_json IS NOT NULL;

UPDATE runtime_config_records
SET payload = rename_notification_json_keys(payload)
WHERE payload IS NOT NULL;

DROP FUNCTION rename_notification_json_keys(jsonb);
