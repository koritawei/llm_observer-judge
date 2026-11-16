-- =============================================================================
-- agent-eval 数据库建表脚本
-- 基于 services/agent-eval/app/data_model/models/ 下的 Tortoise ORM 模型生成
-- 所有表均继承 BaseModel（id UUID PK + created_at + updated_at）
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. trace_pipeline_records — Trace 管线记录
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trace_pipeline_records (
    id              UUID         NOT NULL DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),

    -- 标识
    trace_id        VARCHAR(128) NOT NULL,
    project_id      VARCHAR(128),
    evaluator_name  VARCHAR(512) NOT NULL DEFAULT 'default',

    -- Agent
    agent_name      VARCHAR(128),

    -- 数据来源场景
    scene           VARCHAR(64)  NOT NULL DEFAULT 'non_voice',

    -- 管线状态 (TraceStatus)
    status          VARCHAR(10)  NOT NULL,

    -- 报表统计标量
    owner_group     VARCHAR(128),
    triage_level    VARCHAR(32),
    root_cause      VARCHAR(256),
    notification_job_status VARCHAR(32),

    -- Buckets
    judge_json                                  JSONB,
    attribution_detail_json                     JSONB,
    notification_dispatch_json                          JSONB,
    notification_job_json                               JSONB,
    label_studio_attribution_annotation_json     JSONB,
    label_studio_badcase_judge_annotation_json   JSONB,
    trace_meta_json                             JSONB,

    CONSTRAINT trace_pipeline_records_pkey
        PRIMARY KEY (id),
    CONSTRAINT uq_trace_record_identity
        UNIQUE (trace_id, project_id, agent_name, evaluator_name)
);

CREATE INDEX IF NOT EXISTS idx_trace_pipeline_records_status
    ON trace_pipeline_records (status);
CREATE INDEX IF NOT EXISTS idx_trace_pipeline_records_triage_level
    ON trace_pipeline_records (triage_level);
CREATE INDEX IF NOT EXISTS idx_trace_pipeline_records_root_cause
    ON trace_pipeline_records (root_cause);
CREATE INDEX IF NOT EXISTS idx_trace_pipeline_records_notification_job_status
    ON trace_pipeline_records (notification_job_status);
CREATE INDEX IF NOT EXISTS idx_trace_pipeline_records_owner_group
    ON trace_pipeline_records (owner_group);
CREATE INDEX IF NOT EXISTS idx_trace_pipeline_records_created_at
    ON trace_pipeline_records (created_at);
CREATE INDEX IF NOT EXISTS idx_trace_pipeline_records_scene
    ON trace_pipeline_records (scene);
CREATE INDEX IF NOT EXISTS idx_trace_pipeline_records_project_evaluator
    ON trace_pipeline_records (project_id, evaluator_name);
CREATE INDEX IF NOT EXISTS idx_trace_pipeline_records_agent_name
    ON trace_pipeline_records (agent_name);


-- -----------------------------------------------------------------------------
-- 2. benchmark_runs — Benchmark 运行记录
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id              UUID         NOT NULL DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),

    -- 标识
    run_id          VARCHAR(128) NOT NULL,

    -- 维度
    benchmark_name  VARCHAR(128) NOT NULL,
    service_name    VARCHAR(128) NOT NULL,
    dataset_name    VARCHAR(256) NOT NULL,
    project_id      VARCHAR(128),

    -- 状态 (BenchmarkStatus)
    status          VARCHAR(17)  NOT NULL,
    owner           VARCHAR(256),
    heartbeat_at    TIMESTAMPTZ,

    -- 进度
    total_samples           INT NOT NULL,
    total_evaluators        INT NOT NULL,
    completed_evaluators    INT NOT NULL DEFAULT 0,
    evaluated_items         INT NOT NULL DEFAULT 0,
    failed_items            INT NOT NULL DEFAULT 0,

    -- 结果
    final_score     DOUBLE PRECISION,
    error_message   TEXT,

    -- 软删除标记（BaseModelWithSoftDelete；页面删除单次评测时置位，查询跳过）
    is_deleted      BOOLEAN      NOT NULL DEFAULT FALSE,
    deleted_at      TIMESTAMPTZ,

    -- 时间
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    duration_ms     INT,

    -- Buckets
    metrics_json        JSONB,
    config_json         JSONB,
    report_meta_json    JSONB,

    CONSTRAINT benchmark_runs_pkey
        PRIMARY KEY (id),
    CONSTRAINT uq_benchmark_runs_run_id
        UNIQUE (run_id)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_runs_name_status
    ON benchmark_runs (benchmark_name, status);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_service_status
    ON benchmark_runs (service_name, status);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_status_created
    ON benchmark_runs (status, created_at);


-- -----------------------------------------------------------------------------
-- 5. benchmark_run_items — Benchmark 评测结果明细
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS benchmark_run_items (
    id              UUID         NOT NULL DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),

    run_id          VARCHAR(128) NOT NULL,
    item_id         VARCHAR(128) NOT NULL,
    criterion_id    VARCHAR(128) NOT NULL,
    evaluator_ref   VARCHAR(128) NOT NULL,
    status          VARCHAR(32)  NOT NULL DEFAULT 'running',

    score           DOUBLE PRECISION,
    reason          TEXT,
    sub_scores      JSONB,
    judge_json      JSONB,
    attribution_detail_json JSONB,
    notification_dispatch_json JSONB,
    trace_id        VARCHAR(256),
    source_trace_id VARCHAR(256),
    trace_url       VARCHAR(2048),
    attribution_job_id VARCHAR(256),
    input           JSONB,
    output          JSONB,
    error_message   TEXT,

    CONSTRAINT benchmark_run_items_pkey
        PRIMARY KEY (id),
    CONSTRAINT uq_benchmark_run_items_identity
        UNIQUE (run_id, item_id, criterion_id)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_run_items_run_criterion
    ON benchmark_run_items (run_id, criterion_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_run_items_item
    ON benchmark_run_items (item_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_run_items_attribution_job_id
    ON benchmark_run_items (attribution_job_id);


-- -----------------------------------------------------------------------------
-- 6. dataset 评测表已退出。已有库删除这两张表。
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS dataset_eval_run_items;
DROP TABLE IF EXISTS dataset_eval_runs;


-- =============================================================================
-- 9. ALTER TABLE: 增量迁移（已建库的 schema 对齐）
-- =============================================================================
-- -----------------------------------------------------------------------------
-- 11. ALTER TABLE: benchmark_runs 增 evaluated_items / failed_items
--     （对齐 D12 三阶段写入的 item 级进度追踪；已建库的增量迁移）
-- -----------------------------------------------------------------------------
ALTER TABLE benchmark_runs
    ADD COLUMN IF NOT EXISTS evaluated_items INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS failed_items    INT NOT NULL DEFAULT 0;

-- -----------------------------------------------------------------------------
-- 14. ALTER TABLE: run_items status DEFAULT 改为 running
--     （对齐 D12 三阶段写入：INSERT 基础行 status=running）
-- -----------------------------------------------------------------------------
ALTER TABLE benchmark_run_items
    ALTER COLUMN status SET DEFAULT 'running';

-- -----------------------------------------------------------------------------
-- 15. ALTER TABLE: CHECK 约束对齐 ORM 枚举（已按旧版 schema_pg.sql 建库的增量修复）
--     旧版 CHECK 缺 accepting / attribution_overridden，流式 run 创建
--     （BenchmarkStatus.ACCEPTING）与归因推翻（TriageLevel.ATTRIBUTION_OVERRIDDEN）
--     写入会被约束拒绝。
-- -----------------------------------------------------------------------------
ALTER TABLE benchmark_runs
    DROP CONSTRAINT IF EXISTS ck_benchmark_runs_status,
    ADD CONSTRAINT ck_benchmark_runs_status CHECK (
        status IN (
            'initializing', 'accepting', 'running', 'aggregating',
            'generating_report', 'completed', 'failed'
        )
    );

ALTER TABLE trace_pipeline_records
    DROP CONSTRAINT IF EXISTS ck_trace_records_triage_level,
    ADD CONSTRAINT ck_trace_records_triage_level CHECK (
        triage_level IS NULL OR triage_level IN (
            'good', 'watch', 'mild_bad', 'severe_bad',
            'no_decision', 'attribution_overridden'
        )
    );

-- -----------------------------------------------------------------------------
-- 16. ALTER TABLE: 异步归因 task_id（回调匹配）
-- -----------------------------------------------------------------------------
ALTER TABLE benchmark_run_items
    ADD COLUMN IF NOT EXISTS attribution_job_id VARCHAR(256);
CREATE INDEX IF NOT EXISTS idx_benchmark_run_items_attribution_job_id
    ON benchmark_run_items (attribution_job_id);

-- -----------------------------------------------------------------------------
-- 17. ALTER TABLE: run_items 增加 trace_id 列
--     （数据源原始 trace 标识，judge 回调写入；与 source_trace_id 区分。
--      已建库的增量迁移；新建库由上方 CREATE TABLE 直接包含）
-- -----------------------------------------------------------------------------
ALTER TABLE benchmark_run_items
    ADD COLUMN IF NOT EXISTS trace_id VARCHAR(256);

-- -----------------------------------------------------------------------------
-- 18. ALTER TABLE: run_items 增加 trace_url 列
--     （数据源 trace 页面链接，judge 回调写入；report Case 跳转用。
--      已建库的增量迁移；新建库由上方 CREATE TABLE 直接包含）
-- -----------------------------------------------------------------------------
ALTER TABLE benchmark_run_items
    ADD COLUMN IF NOT EXISTS trace_url VARCHAR(2048);

-- -----------------------------------------------------------------------------
-- 19. ALTER TABLE: run_items 增加 notification_dispatch_json 列
--     （Aone 工单推送 bucket，归因回调推送后写入；对齐
--      BenchmarkRunItem 的 notification_dispatch_json 字段。
--      已建库的增量迁移；新建库由上方 CREATE TABLE 直接包含）
-- -----------------------------------------------------------------------------
ALTER TABLE benchmark_run_items
    ADD COLUMN IF NOT EXISTS notification_dispatch_json JSONB;

-- -----------------------------------------------------------------------------
-- 20. cluster_report_jobs — 持久化报告任务
--     仅作为 DMS 变更单附件执行；应用启动不会自动改正式库。
-- 恢复扫描只按任务状态与租约过期时间筛选。
-- source_batch_tags 参与 idempotency_key 计算；作业查重走该哈希唯一键，
-- 不执行 JSON 数组包含查询，故不创建 GIN 索引。
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cluster_report_jobs (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    job_id UUID NOT NULL DEFAULT gen_random_uuid(),
    environment VARCHAR(32) NOT NULL,
    evaluator_name VARCHAR(128) NOT NULL,
    job_type VARCHAR(32) NOT NULL,
    source_batch_tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    idempotency_key VARCHAR(128) NOT NULL,

    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    phase VARCHAR(32) NOT NULL DEFAULT '',
    attempt INT NOT NULL DEFAULT 0,
    lease_owner VARCHAR(128) NOT NULL DEFAULT '',
    lease_expire_at TIMESTAMPTZ,

    job_request JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifacts JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    dispatch JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT NOT NULL DEFAULT '',

    CONSTRAINT cluster_report_jobs_pkey PRIMARY KEY (id),
    CONSTRAINT uq_cluster_report_jobs_job_id UNIQUE (job_id),
    CONSTRAINT uq_cluster_report_jobs_request
        UNIQUE (environment, evaluator_name, job_type, idempotency_key),
    CONSTRAINT ck_cluster_report_jobs_type
        CHECK (job_type IN ('normal', 'stability', 'performance')),
    CONSTRAINT ck_cluster_report_jobs_status
        CHECK (status IN ('pending', 'running', 'completed', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_cluster_report_jobs_recovery
    ON cluster_report_jobs (status, lease_expire_at);

-- -----------------------------------------------------------------------------
-- 21. benchmark_kafka_tasks — streaming 任务声明态镜像
--     （RemoteConfig agent-eval.benchmark.streaming 为唯一真相源；本表为 reconciler
--      每轮对账成功后的声明态镜像快照，供任务清单查询 / 审计 / P6 页面化复用。
--      软删审计由 RemoteConfig 发布历史承载：配置移除条目 = 删除行）
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS benchmark_kafka_tasks (
    id              UUID         NOT NULL DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),

    task_id         VARCHAR(128) NOT NULL,
    source          VARCHAR(32)  NOT NULL DEFAULT 'remote_config',
    state           VARCHAR(16)  NOT NULL DEFAULT 'running',
    topics          JSONB        NOT NULL DEFAULT '[]'::jsonb,
    group_id        VARCHAR(256) NOT NULL,
    judge_class     VARCHAR(256),
    attribution_class VARCHAR(256),
    rubric_class    VARCHAR(256),
    sampling        DOUBLE PRECISION,
    daily_report_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    shadow          BOOLEAN      NOT NULL DEFAULT FALSE,
    spec_fingerprint VARCHAR(64),
    last_reconciled_at TIMESTAMPTZ,

    CONSTRAINT benchmark_kafka_tasks_pkey
        PRIMARY KEY (id),
    CONSTRAINT uq_benchmark_kafka_tasks_task_id
        UNIQUE (task_id)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_kafka_tasks_state
    ON benchmark_kafka_tasks (state);

-- -----------------------------------------------------------------------------
-- 22. benchmark_report_providers — 业务外部报告服务注册表
--     （resolve_report_provider 解析优先级最高通道：DB > RemoteConfig 种子配置 >
--      统一插件 bindings > platform 默认；由 /v1/benchmarks/report-providers
--      CRUD API 管理，业务方运行时自助注册，无需发布 RemoteConfig）
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS benchmark_report_providers (
    id              UUID         NOT NULL DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),

    benchmark_name  VARCHAR(128) NOT NULL,
    provider        VARCHAR(32)  NOT NULL DEFAULT 'external_http',
    endpoint_url    VARCHAR(1024) NOT NULL,
    bearer_token    VARCHAR(1024),
    timeout_ms      INTEGER      NOT NULL DEFAULT 120000,
    fallback_to_platform BOOLEAN NOT NULL DEFAULT TRUE,
    enabled         BOOLEAN      NOT NULL DEFAULT TRUE,
    description     VARCHAR(512),

    CONSTRAINT benchmark_report_providers_pkey
        PRIMARY KEY (id),
    CONSTRAINT uq_benchmark_report_providers_name
        UNIQUE (benchmark_name)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_report_providers_enabled
    ON benchmark_report_providers (enabled);

-- -----------------------------------------------------------------------------
-- 22. runtime_config_records — 页面配置唯一新增表
--     信封列固定。新 kind 只增加 kind 取值和 payload 形状，不新建表、不加列。
--     已用 kind：evaluator、odps_job、benchmark_agent、benchmark_dataset、eval_task。
--     预留：scheduled_task、dataset_task、kafka_task。控制台 kind 不进入 benchmark 执行。
--     scope_key 前缀：project: / benchmark: / agent: / global。
--     种类特有字段放 payload；GIN 供偶发包含查询。确认高频过滤后再升成列。
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runtime_config_records (
    id              UUID         NOT NULL DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),

    kind            VARCHAR(64)  NOT NULL,
    record_key      VARCHAR(512) NOT NULL,
    schema_version  INTEGER      NOT NULL DEFAULT 1,
    revision        INTEGER      NOT NULL DEFAULT 1,
    scope_key       VARCHAR(256) NOT NULL DEFAULT 'global',
    enabled         BOOLEAN      NOT NULL DEFAULT TRUE,
    deleted_at      TIMESTAMPTZ,
    payload         JSONB        NOT NULL,

    CONSTRAINT runtime_config_records_pkey
        PRIMARY KEY (id),
    CONSTRAINT uq_runtime_config_records_key
        UNIQUE (record_key)
);

CREATE INDEX IF NOT EXISTS idx_runtime_config_records_scope
    ON runtime_config_records (kind, scope_key)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_runtime_config_records_enabled
    ON runtime_config_records (kind, enabled)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_runtime_config_records_payload
    ON runtime_config_records USING GIN (payload jsonb_path_ops);

-- -----------------------------------------------------------------------------
-- 23. ALTER TABLE: benchmark_runs 增软删除标记（BaseModelWithSoftDelete）
--     （页面逐期删除单次评测只打标记不删行；所有页面查询跳过已标记行。
--     已建库的增量迁移；新建库由上方 CREATE TABLE 直接包含）
-- -----------------------------------------------------------------------------
ALTER TABLE benchmark_runs
    ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
