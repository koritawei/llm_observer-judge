# agent-eval 运维脚本

工作目录一律是 `services/agent-eval`。有同名 `.sh` 的入口必须走 shell 包装，不要裸跑 `python3`。包装脚本会在线上 source `APP_HOME/base/bin/setenv.sh` 并补齐 `PYTHONPATH`。

## 严格限制

1. **只有 code owner 可以执行本目录任何脚本。** owner 以仓库根目录 [`CODEOWNERS`](../../../CODEOWNERS) 中 `services/agent-eval/**` 为准：`@yisou.gj`、`@shengzhiqiang.szq`、`@weiwei.qw`、`@miaoxujie.mxj`。
2. 非 owner 不得执行、不得代跑、不得把带真实连接串或密钥的命令交给他人执行。
3. 写库、删数、清队列、建 Aone 工单、上传 OSS、对共享环境触发评测，必须由 owner 本人操作。未加 `--execute` 的 dry-run 只打印计划，不代表已经生效。
4. 单次删除或插入必须少于 10000 行。达到上限时脚本会拒绝，需分批或走变更单。
5. 禁止把数据库口令、OSS AccessKey、Aone token 写进本目录、提交记录或聊天记录。命令行参数里的连接串只留在 owner 自己的终端。
6. `_bootstrap.py`、`_bootstrap_quiet.py` 是库，禁止当入口执行。`schema_pg.sql` 只建 `trace_pipeline_records`，不是全库迁移脚本。

## 数据迁移与库结构

### `backfill_agent_scope.sh`

作用：为存量 `kind=benchmark_agent` 注册回填 `payload.agentScope`，并同步
`scope_key=agent:{scope}`。覆盖**普通注册 + 定制注册**；仅处理缺
`agentScope` 的活跃行。默认归属 `amap-agent`（历史控制台唯一页签）。

用法：

```bash
bash scripts/backfill_agent_scope.sh
bash scripts/backfill_agent_scope.sh --apply
bash scripts/backfill_agent_scope.sh --apply --default-scope amap-agent
```

严格限制：默认 dry-run。`--apply` / `--execute` 才写库。单次 pending ≥ 10000
拒绝执行。禁止把真实 `AGENT_EVAL_DATABASE_URL` 写入仓库或聊天。name 仍受 L1
白名单约束：回填后不在白名单的 name 不会出现在对应评测对象列表。线上须能解析
`agentgraph`（包装脚本会查 `$BAZEL_WORKSPACE_HOME/libs/agent_graph/src`，缺则
`FATAL`）；勿绕过 `.sh` 直接裸跑 `python3`。

### `migrate_benchmark_tables.sh`

作用：把 bullMQ 库的 `benchmark_runs`、`benchmark_run_items` 迁入 main 库的 `benchmark_runs`、`benchmark_run_items`。已有 `run_id`、明细键或 `id` 自动跳过，只插入 diff，不更新已有行。

用法：

```bash
bash scripts/migrate_benchmark_tables.sh \
  --source-url 'postgresql://USER:PASSWORD@bullmq-host:5432/agent_eval' \
  --target-url 'postgresql://USER:PASSWORD@main-host:5432/agent_eval'
bash scripts/migrate_benchmark_tables.sh \
  --source-url 'postgresql://USER:PASSWORD@bullmq-host:5432/agent_eval' \
  --target-url 'postgresql://USER:PASSWORD@main-host:5432/agent_eval' \
  --execute
```

严格限制：`--source-url` 与 `--target-url` 都必填，且必须指向不同职责的库。默认 dry-run。禁止对仍在写入的目标库执行 `--execute`，除非 owner 已确认停写。`benchmark_kafka_tasks` 与 `benchmark_report_providers` 不在迁移范围内。

### `reset_schema.sh`

作用：按当前 ORM 模型 `DROP TABLE ... CASCADE` 后重建全部表、索引和约束。

用法：

```bash
bash scripts/reset_schema.sh
bash scripts/reset_schema.sh --execute
```

严格限制：默认 dry-run。`--execute` 会清空该连接串上的全部业务表。禁止对预发或生产库执行，除非 owner 已停写并完成备份。

### `schema_pg.sql`

作用：仅包含 `trace_pipeline_records` 的 PostgreSQL 建表语句，供对照模型。

用法：`psql -h <host> -U <user> -d <db> -f scripts/schema_pg.sql`

严格限制：`CREATE TABLE IF NOT EXISTS` 不会改已有列。禁止把它当成全库 schema 或数据迁移脚本。

### `delete_run_data.sh`

作用：按 `run_id` 删除 `benchmark_run_items` 与 `benchmark_runs`。不清队列、Redis 或 inflight 槽位，也不重新触发。

用法：

```bash
bash scripts/delete_run_data.sh --run-id ieval_external-20260819
bash scripts/delete_run_data.sh --run-id ieval_external-20260819 --execute
bash scripts/delete_run_data.sh --run-id ieval_external-20260819 --execute --force
```

严格限制：默认 dry-run。活动态 run 必须先 `terminate_run.sh --execute`，或由 owner 显式加 `--force`。行数达到 10000 时拒绝执行。

### `table_ops.sh`

作用：表级运维四合一——① `delete-rows` 删除指定表、指定 ID 的行（`--id-column` 可配，缺省 `id`；ID 支持正整数主键与字符串业务 ID 如 `run_id` 的 UUID）；② `truncate-table` 清空表数据但保留表结构（需 `--execute --force` 双确认）；③ `drop-table` 删除整张表（需 `--execute --force` 双确认）；④ `create-table` 执行建表 SQL 文件（仅放行 CREATE TABLE/INDEX/TYPE 与 COMMENT ON，拒绝 DROP/DELETE/UPDATE 等高危语句）。

用法：

```bash
# 删指定表指定 ID 的行（先 dry-run 预览 found/missing，再加 --execute）
bash scripts/table_ops.sh delete-rows --table runtime_config_records --ids 1,2,3
bash scripts/table_ops.sh delete-rows --table runtime_config_records --ids 1,2,3 --execute

# 字符串业务 ID（如 benchmark_runs.run_id 的 UUID）：指定 --id-column
bash scripts/table_ops.sh delete-rows --table benchmark_runs \
  --id-column run_id --ids 4bc42a6a-610e-475c-96ae-ad8056607d99
bash scripts/table_ops.sh delete-rows --table benchmark_runs \
  --id-column run_id --ids 4bc42a6a-610e-475c-96ae-ad8056607d99 --execute

# 清空表数据但保留表结构（数据不可恢复，双确认）
bash scripts/table_ops.sh truncate-table --table tmp_xxx --execute --force

# 删整张表（不可恢复，双确认）
bash scripts/table_ops.sh drop-table --table tmp_xxx --execute --force

# 按建表 SQL 新建表（支持多语句分号分隔；或 stdin）
bash scripts/table_ops.sh create-table --sql-file ./sql/tmp_table.sql --execute
cat tmp.sql | bash scripts/table_ops.sh create-table --sql-file - --execute

# 给已有表加列（软删除标记等加列型 schema 变更；列已存在自动跳过）
bash scripts/table_ops.sh add-column --table benchmark_runs \
  --column deleted_at --type TIMESTAMPTZ
bash scripts/table_ops.sh add-column --table benchmark_runs \
  --column deleted_at --type TIMESTAMPTZ --execute
```

`add-column` 只放行 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS <列> <类型>` 一种形态：表/列名与列类型均过白名单校验，不带 DEFAULT/NOT NULL 等子句；只加可空无默认列，在线执行不锁表。

严格限制：默认 dry-run。DELETE 恒带 WHERE（按 ID 列过滤），单次 < 10000 行（DB-RDB-003）；表名/列名过标识符白名单正则；`truncate-table` 与 `drop-table` 需 `--execute --force` 且优先建议走 DMS 变更单。

### `reassign_benchmark_run.sh`

作用：用 `-a1` run 覆盖同日正式 `run_id`（删除 target，再把 source 改名为 target）。

用法：

```bash
bash scripts/reassign_benchmark_run.sh
bash scripts/reassign_benchmark_run.sh --execute
bash scripts/reassign_benchmark_run.sh \
  --source ieval_external-20260921-a1 \
  --target ieval_external-20260921 --execute
```

严格限制：默认 dry-run。`--execute` 会删除 target 上已有评测结果，不可自动恢复。

## 重跑、终止与槽位

### `rerun.sh`

作用：完全重跑一个 run：删除全部 items、清队列，再重新触发。

用法：

```bash
bash scripts/rerun.sh --run-id ieval_external-20260819
bash scripts/rerun.sh --run-id ieval_external-20260819 --execute
```

严格限制：默认 dry-run。`--execute` 丢弃该 run 已有明细并重新入队。禁止在该 run 仍有在途回调时执行。

### `rerun_failed.sh`

作用：只重跑 `status=failed` 的 items：删除失败行、修正计数器、重新入队。成功行保留。

用法：

```bash
bash scripts/rerun_failed.sh --run-id ieval_external-20260819
bash scripts/rerun_failed.sh --run-id ieval_external-20260819 --execute
```

严格限制：默认 dry-run。只允许针对已经终态失败的 item，不得用来清整次 run。

### `rerun_stuck_external_finalize.sh`

作用：处理卡在 `aggregating` / `generating_report` 的 ieval external finalize。可列出、重跑 finalize、标失败或清 inflight 槽位。

用法：

```bash
bash scripts/rerun_stuck_external_finalize.sh --list
bash scripts/rerun_stuck_external_finalize.sh --run-id ieval_external-20260812-03 --finalize --dry-run
bash scripts/rerun_stuck_external_finalize.sh --run-id ieval_external-20260812-03 --finalize
bash scripts/rerun_stuck_external_finalize.sh --run-id ieval_external-20260812-03 --finalize --force
bash scripts/rerun_stuck_external_finalize.sh --run-id ieval_external-20260812-03 --mark-failed
bash scripts/rerun_stuck_external_finalize.sh --run-id ieval_external-20260812-03 --clear-inflight-slots --dry-run
```

严格限制：`--mark-failed` 与 `--clear-inflight-slots` 会改运行状态或槽位。先 `--list` 或 `--dry-run`，确认 run 确实卡住后再写。

### `terminate_run.sh`

作用：把 run 标为终止，并释放 judge 与 attribution 的全部槽位。

用法：

```bash
bash scripts/terminate_run.sh --run-id ieval_external-20260812-34
bash scripts/terminate_run.sh --run-id ieval_external-20260812-34 --execute
bash scripts/terminate_run.sh --run-id ieval_external-20260812-34 --execute --reason "ops: stuck"
```

严格限制：默认 dry-run。终止后在途回调不应再写回该 run。需要删行时再单独跑 `delete_run_data.sh`。

### `clear_attribution_running_slots.sh`

作用：清除卡住的 attribution-running `pending_callback` 及其 inflight 槽位。

用法：

```bash
bash scripts/clear_attribution_running_slots.sh --job-id <job-id>
bash scripts/clear_attribution_running_slots.sh --job-id <job-id> --execute
```

严格限制：必须显式传入 `--job-id`，禁止不带 job 清全量槽位。默认 dry-run。

## 报表、工单与触发

### `regenerate_report.sh`

作用：为 case 已跑完的 external run 生成或重生成日报。

用法：

```bash
bash scripts/regenerate_report.sh --list
bash scripts/regenerate_report.sh --run-id ieval_external-20260814-03 --dry-run
bash scripts/regenerate_report.sh --run-id ieval_external-20260814-03
```

严格限制：`--list` 与 `--dry-run` 不发布报告。不带 `--dry-run` 会覆盖该 run 已有报告产物。

### `trace_daily_report.sh` / `trace_daily_report_dryrun.py`

作用：按 trace 日调度配置跑 T+1 日报。shell 是线上入口，Python 是同一条链路的 dry-run 实现。

用法：

```bash
bash scripts/trace_daily_report.sh
bash scripts/trace_daily_report.sh --name amap_agent_daily --date 20260928
bash scripts/trace_daily_report.sh --name amap_agent_daily --execute
bash scripts/trace_daily_report.sh --name amap_agent_daily --date 20260929 --execute --force
```

严格限制：不加 `--execute` 只做 dry-run。`--force` 会忽略“当日已出报”一类保护，只允许 owner 在确认需要重出时使用。

### `dispatch_aone_for_run.sh`

作用：对指定 run 里已完成归因的 case 批量补推 Aone 工单。

用法：

```bash
bash scripts/dispatch_notification_for_run.sh --run-id ieval_external-20260820-01 --dry-run
bash scripts/dispatch_notification_for_run.sh --run-id ieval_external-20260820-01
```

严格限制：不带 `--dry-run` 会创建真实工单。必须先 dry-run，核对条数后再由 owner 执行正式命令。禁止重复补推已经派过单的 run。

### `trigger_benchmark.sh`

作用：向 agent-eval 提交一次离线 benchmark。默认 POST 后立即退出；`POLL=true` 才等到终态。`LIST_ONLY=true` 只列出已注册 benchmark。

用法：

```bash
BENCHMARK_NAME="example_benchmark" bash scripts/trigger_benchmark.sh
POLL=true BENCHMARK_NAME="example_benchmark" bash scripts/trigger_benchmark.sh
LIST_ONLY=true bash scripts/trigger_benchmark.sh
```

严格限制：默认打到预发 `https://pre-aiagent-eval.amap-inc.com`。改 `AGENT_EVAL_URL` 指向生产前，owner 必须确认采样量和并发。服务端同一时间只跑一个 benchmark。

### `trigger_pipeline_benchmark.sh`

作用：触发 pipeline benchmark，只聚合已有 Kafka 评分，不调 LLM、不拉 Langfuse dataset。

用法：

```bash
BENCHMARK_NAME="example_benchmark" \
DATE_START="2026-08-04" DATE_END="2026-08-05" \
bash scripts/trigger_pipeline_benchmark.sh
LIST_ONLY=true bash scripts/trigger_pipeline_benchmark.sh
```

严格限制：与常规 benchmark 共用进程内锁。日期窗必须由 owner 指定，禁止用空日期扫全历史。

## OSS / ODPS

### `export_odps_to_oss.sh`

作用：按业务 job 把 ODPS 抽样导出成 CSV，再上传 OSS。可补跑指定分区。

用法：

```bash
bash scripts/export_odps_to_oss.sh
bash scripts/export_odps_to_oss.sh --ds 20260907
bash scripts/export_odps_to_oss.sh --no-upload
bash scripts/export_odps_to_oss.sh --dry-run
bash scripts/export_odps_to_oss.sh --oss-ds 20260909
bash scripts/export_odps_to_oss.sh --job search-agent-overview --no-upload
```

严格限制：`--dry-run` 不连 ODPS。不上 `--no-upload` 时会写 OSS。禁止把 AK/SK 写进命令后提交。短命脚本不要靠服务启动补偿去导出。

### `upload_oss_file.sh`

作用：把本地文件上传到 OSS，object key 前缀由参数指定。

用法：

```bash
bash scripts/upload_oss_file.sh --prefix misc/dumps /tmp/a.csv
bash scripts/upload_oss_file.sh --object-key misc/readme.html /tmp/readme.html --sign
bash scripts/upload_oss_file.sh --prefix misc/dumps --dry-run /tmp/a.csv
```

严格限制：先 `--dry-run` 核对 key。`--sign` 会生成可访问链接，链接不得发到 owner 范围以外。禁止上传密钥、库备份或未授权数据。

### `dataworks_pyodps3_odps_export.py`

作用：DataWorks「PyODPS 3」节点正文。用工作空间已绑定的部门账号读 ODPS，写出 CSV 并上传 OSS。

用法：把脚本贴进 DataWorks 节点。调度参数使用 `ds`、`limit`、`project_id`、`oss_prefix`、`oss_access_id`、`oss_access_key`、`oss_endpoint`、`oss_bucket`。节点里已经有 `o` / `odps` / `args`，不要再手建 ODPS 入口。

严格限制：禁止在本仓库里当服务入口执行。禁止再写一套 `ODPS(access_id, ...)`。真实 AccessKey 只放在 DataWorks 调度参数，禁止回写到 git。

## 只读检查与本地工具

### `sidecar.sh`

作用：点查一条 sidecar trace（Lindorm/OSS → 过滤 → 四个 trace tool）。不订阅 MetaQ，不跑 judge。

用法：

```bash
bash scripts/sidecar.sh --trace-id <id>
bash scripts/sidecar.sh --trace-id <id> --project-id <pid>
bash scripts/sidecar.sh --trace-id <id> --project-id <pid> --print-notify
```

严格限制：只读。`--print-notify` 会把通知正文打到终端，禁止转发含用户内容的输出。

### `smoke_trace_pipeline.py`

作用：对一条 `trace_id` 跑 judge → triage → Langfuse resolve → attribution 冒烟。

用法：

```bash
.venv/bin/python scripts/smoke_trace_pipeline.py \
  --trace-id <trace-id> --project-id vehicle
```

严格限制：结果会写入 `AGENT_EVAL_DATABASE_URL` 指向的库，重复跑会按唯一键覆盖。Aone 强制 `dry_run=True`，不会真建单。禁止指向生产库。

### `check_streaming_config_alignment.py`

作用：读 RemoteConfig，校验 streaming、trace 日报和 eval config 的 group、rubric、evaluator 是否一致。失败时 exit 1。

用法：`python scripts/check_streaming_config_alignment.py`

严格限制：只读配置，不改 RemoteConfig。CI 可以调用；人工重跑仍须 owner 执行。

### `run_unit_tests.py`

作用：先跑 SQLite 单测，再在新进程里跑 PG 单测，避免 Tortoise 方言状态串扰。

用法：`.venv/bin/python scripts/run_unit_tests.py`

严格限制：PG 段依赖调用方环境里的 `PG_*`。没有 PG 环境时对应用例会 skip。禁止把生产库地址填进 `PG_*`。

### `demo_three_state_upsert.py`

作用：用本地 SQLite 演示 TraceRecord 三态 upsert，每步打印库内状态。

用法：在 `services/agent-eval` 下用项目虚拟环境执行 `python scripts/demo_three_state_upsert.py`。

严格限制：只允许打到本地 SQLite。禁止改脚本去连共享 PostgreSQL。

## 内部模块

| 文件                    | 作用                                                                      | 限制                                    |
| ----------------------- | ------------------------------------------------------------------------- | --------------------------------------- |
| `_bootstrap.py`       | 给运维脚本注入 `sys.path`、加载 `.env`、默认关闭 MetaQ                | 只能被其它脚本 `import`，禁止直接执行 |
| `_bootstrap_quiet.py` | 在 `_bootstrap` 上再关闭 healer、日调度、ODPS、streaming，并强制 `SIDECAR_TRACE_ENABLED=false`（防 CMMS Abort） | 同上。需要安静 lifespan 的脚本才导入    |
