CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id text PRIMARY KEY,
    benchmark_name text NOT NULL,
    status text NOT NULL,
    total_samples integer NOT NULL,
    total_evaluators integer NOT NULL,
    evaluated_items integer NOT NULL DEFAULT 0,
    failed_items integer NOT NULL DEFAULT 0,
    judge_expected_count integer NOT NULL DEFAULT 0,
    judge_received_count integer NOT NULL DEFAULT 0,
    attribution_expected_count integer NOT NULL DEFAULT 0,
    attribution_received_count integer NOT NULL DEFAULT 0,
    input_sealed boolean NOT NULL DEFAULT false,
    attribution_decisions_sealed boolean NOT NULL DEFAULT false,
    required_outbox_delivered boolean NOT NULL DEFAULT false,
    completion_extra_ready boolean NOT NULL DEFAULT false,
    judge_wait_purged boolean NOT NULL DEFAULT false,
    admission_epoch integer NOT NULL DEFAULT 0,
    admission_closed boolean NOT NULL DEFAULT false,
    stop_requested_at timestamptz,
    stop_target_count integer,
    stop_admit_snapshot jsonb,
    stop_progress_keys jsonb,
    completion_model text NOT NULL,
    config_json jsonb NOT NULL,
    metrics_json jsonb,
    report_meta_json jsonb,
    final_score double precision,
    owner text,
    heartbeat_at timestamptz,
    error_message text,
    parent_run_id text,
    idempotency_key text UNIQUE,
    started_at timestamptz,
    finished_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS benchmark_run_items (
    run_id text NOT NULL,
    item_id text NOT NULL,
    criterion_id text NOT NULL,
    evaluator_ref text NOT NULL DEFAULT '',
    status text NOT NULL,
    score double precision,
    reason text,
    judge_json jsonb,
    attribution_detail_json jsonb,
    dispatch_json jsonb,
    source_trace_id text,
    input jsonb,
    output jsonb,
    error_message text,
    external_job_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, item_id, criterion_id)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_run_items_status
    ON benchmark_run_items (run_id, status);

CREATE TABLE IF NOT EXISTS input_manifest (
    run_id text NOT NULL,
    item_id text NOT NULL,
    payload jsonb NOT NULL,
    checksum text NOT NULL,
    PRIMARY KEY (run_id, item_id)
);

CREATE TABLE IF NOT EXISTS admissions (
    run_id text NOT NULL,
    item_id text NOT NULL,
    epoch integer NOT NULL,
    PRIMARY KEY (run_id, item_id)
);

CREATE TABLE IF NOT EXISTS phase_cursors (
    run_id text NOT NULL,
    item_id text NOT NULL,
    phase_id text NOT NULL,
    output jsonb,
    PRIMARY KEY (run_id, item_id, phase_id)
);

CREATE TABLE IF NOT EXISTS run_progress_events (
    run_id text NOT NULL,
    stage text NOT NULL,
    dedupe_key text NOT NULL,
    PRIMARY KEY (run_id, stage, dedupe_key)
);

CREATE TABLE IF NOT EXISTS attribution_decisions (
    run_id text NOT NULL,
    item_id text NOT NULL,
    criterion_id text NOT NULL,
    detail jsonb,
    PRIMARY KEY (run_id, item_id, criterion_id)
);

CREATE TABLE IF NOT EXISTS outbox (
    run_id text NOT NULL,
    event_kind text NOT NULL,
    dedupe_key text NOT NULL,
    status text NOT NULL,
    payload jsonb,
    PRIMARY KEY (run_id, event_kind, dedupe_key)
);

CREATE TABLE IF NOT EXISTS submission_intents (
    idempotency_key text PRIMARY KEY,
    run_id text NOT NULL,
    item_id text NOT NULL,
    phase_id text NOT NULL,
    attempt_epoch integer NOT NULL,
    lease_node_id text,
    lease_cluster_id text,
    status text NOT NULL,
    request jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lease_ledger (
    lease_id text PRIMARY KEY,
    capacity_key text NOT NULL,
    run_id text NOT NULL DEFAULT '',
    item_id text NOT NULL DEFAULT '',
    abandoned boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS trace_pipeline_records (
    id bigserial PRIMARY KEY,
    trace_id text NOT NULL,
    project_id text NOT NULL,
    agent_name text NOT NULL,
    evaluator_name text NOT NULL,
    status text NOT NULL,
    score double precision,
    trace_meta_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (trace_id, project_id, agent_name, evaluator_name)
);

CREATE TABLE IF NOT EXISTS agent_registrations (
    registration_id text PRIMARY KEY,
    name text NOT NULL,
    business text NOT NULL,
    variant_key text NOT NULL,
    agent_scope text NOT NULL,
    version integer NOT NULL,
    internal_agent_name text NOT NULL,
    is_default boolean NOT NULL DEFAULT false,
    customization jsonb NOT NULL,
    content_ref text,
    content_hash text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_benchmark_runs_status
    ON benchmark_runs (status, created_at);

CREATE TABLE IF NOT EXISTS eval_experiments (
    experiment_id text PRIMARY KEY,
    evaluation_stage text NOT NULL,
    status text NOT NULL,
    contract_json jsonb NOT NULL,
    gate_policy_json jsonb NOT NULL,
    bundle_json jsonb,
    stage_outcome text,
    release_authorization text,
    run_terminal text,
    source_run_id text,
    candidate_hash text,
    conflict_fingerprint jsonb NOT NULL DEFAULT '[]',
    pending_epoch integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_release_events (
    event_id text PRIMARY KEY,
    experiment_id text NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evolution_requests (
    source_key text PRIMARY KEY,
    tenant_id text NOT NULL,
    run_id text NOT NULL,
    report_revision integer NOT NULL,
    status text NOT NULL,
    reason text,
    campaign_id text,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evolution_campaigns (
    campaign_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    source_run_id text NOT NULL,
    status text NOT NULL,
    execution_epoch integer NOT NULL DEFAULT 0,
    contract_hash text,
    experiment_id text,
    budget_json jsonb NOT NULL DEFAULT '{}',
    diagnosis_json jsonb,
    plan_json jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evolution_contracts (
    contract_hash text PRIMARY KEY,
    campaign_id text NOT NULL,
    manifest jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evolution_heldout (
    campaign_id text NOT NULL,
    item_id text NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (campaign_id, item_id)
);

CREATE TABLE IF NOT EXISTS evolution_steps (
    step_id text PRIMARY KEY,
    campaign_id text NOT NULL,
    step_kind text NOT NULL,
    scope_key text NOT NULL,
    status text NOT NULL,
    epoch integer NOT NULL DEFAULT 0,
    output_hash text,
    output jsonb,
    UNIQUE (campaign_id, step_kind, scope_key)
);

CREATE TABLE IF NOT EXISTS evolution_candidates (
    candidate_id text PRIMARY KEY,
    campaign_id text NOT NULL,
    artifact_hash text NOT NULL,
    artifact_kind text NOT NULL,
    status text NOT NULL,
    diff_text text NOT NULL,
    proposed_outputs jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (campaign_id, artifact_hash)
);

CREATE TABLE IF NOT EXISTS evolution_trials (
    trial_id text PRIMARY KEY,
    campaign_id text NOT NULL,
    candidate_id text NOT NULL,
    benchmark_run_id text,
    stage text NOT NULL,
    status text NOT NULL
);

CREATE TABLE IF NOT EXISTS evolution_confirmations (
    confirmation_key text PRIMARY KEY,
    campaign_id text NOT NULL,
    heldout_manifest_hash text NOT NULL,
    attempt_ordinal integer NOT NULL,
    result_json jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS evolution_releases (
    release_id text PRIMARY KEY,
    campaign_id text NOT NULL,
    artifact_hash text NOT NULL,
    status text NOT NULL,
    epoch integer NOT NULL DEFAULT 0,
    approval_ref text,
    rollback_ref text
);
