-- Support the evaluator[@registrationId]@streamInstanceId composite identity key.
-- Widen the column only; do not delete or rewrite existing data. Execute via a DMS change ticket.
ALTER TABLE trace_pipeline_records
    ALTER COLUMN evaluator_name TYPE VARCHAR(512);
