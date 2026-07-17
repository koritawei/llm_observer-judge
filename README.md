# LLMJudge

LLM 可观测与 LLM-as-a-Judge 平台：接入 Trace/Span → 采样落库 → 路由评估 → Judge 打分 → 坏样本归因（主链路）→ 人工标注（旁路），并提供 Dashboard 与 Prompt A/B 实验。

## 能力概览

| 能力 | 说明 |
|------|------|
| 摄取 | Gateway `POST /api/v1/ingest`，SDK `@observe` |
| 评估 | 多 Judge DAG（并行/依赖/条件），HTTP 兼容 OpenAI（如 DeepSeek） |
| 坏样本 | 低于阈值触发；**归因走主链路**；好样本跳过归因 |
| 标注 | 按路由规则触发，**旁路**，不阻断主链路 |
| 看板 | Trace 列表/详情、执行图与时间线同源状态、标注队列、统计、A/B、运维 |
| Prompt | 版本管理 + A/B 分流（`trace_id` hash） |
| 运行时 | 单进程 DevStack（内存 MQ）或多进程（Redis Streams + SQL） |

## 流水线

```text
SDK / HTTP Ingest
    → Sampling → Index → Write → Outbox
    → Dispatch（唯一生产 eval_jobs）
    → Eval（Judge DAG → Score）
         ├─ 坏样本 → Attribution（主链路）
         └─ 规则触发 → Annotation（旁路）
```

硬错误（`infra_error`）会 fail-stop 后续评估节点；`model_error` / 低分坏样本不阻断整图。

## 环境要求

- Python **3.12+**
- Node.js 18+（仅跑前端时需要）
- 可选：Docker（Postgres / Redis / Compose）

## 安装

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 可选：Postgres / Redis 驱动
pip install -e ".[postgres,redis,dev]"

cp .env.example .env               # 填入密钥后保存（.env 已 gitignore）
pytest
```

`llmj-stack` / `llmj-worker` / `llmj-db` 启动时会自动加载仓库根目录 `.env`。

## 配置（`.env`）

常用变量：

| 变量 | 含义 | 本机常用默认 |
|------|------|----------------|
| `LLMJUDGE_API_KEYS` | `key:tenant` 列表 | `dev-key:tenant-dev` |
| `LLMJUDGE_PERSISTENCE` | `memory` / `sql` | `memory` |
| `LLMJUDGE_DATABASE_URL` | SQLAlchemy 异步 URL | Postgres 或 SQLite |
| `LLMJUDGE_MQ` | `memory` / `redis`（`rocketmq` 未实现） | `memory` |
| `LLMJUDGE_REDIS_URL` | Redis | `redis://127.0.0.1:6379/0` |
| `LLMJUDGE_DISPATCH_CONFIG` | 路由 YAML | `configs/dispatch.yaml` |
| `LLMJUDGE_FORCE_SAMPLE` | 强制采样 | `1` |
| `LLMJUDGE_HOST` / `LLMJUDGE_PORT` | 监听 | `0.0.0.0` / `8080` |
| `LLMJUDGE_JUDGE_BASE_URL` | Judge HTTP 基址 | 如 `https://api.deepseek.com` |
| `LLMJUDGE_JUDGE_API_KEY` | Judge API Key | （必填才走 HTTP Judge） |
| `LLMJUDGE_JUDGE_MODEL_NAME` | 模型名 | 如 `deepseek-v4-flash` |
| `LLMJUDGE_JUDGE_ENFORCE_SSRF` | SSRF 校验 | 本机代理可设 `0` |

Compose 容器内主机名用 `*_COMPOSE` 变量（如 `LLMJUDGE_DATABASE_URL_COMPOSE`、`LLMJUDGE_MQ_COMPOSE`）。  
也可使用别名 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`（未设置正式 `LLMJUDGE_JUDGE_*` 时生效）。

## 本地最快跑通

### 1. 后端（单进程全链路）

```bash
source .venv/bin/activate
llmj-stack
```

- API：`http://127.0.0.1:8080`
- 健康检查：`/healthz`、`/readyz`、`/stackz`

挂载子应用：

| 前缀 | 用途 |
|------|------|
| `/` | Gateway 摄取 / 鉴权 / 遥测 |
| `/dashboard` | Trace、统计、WebSocket |
| `/annotation` | 标注 claim / complete |
| `/benchmark` | 批次评测 |
| `/prompt` | Prompt 版本与 A/B 实验 |

默认 API Key：`dev-key`，租户：`tenant-dev`。

### 2. 前端 Dashboard

```bash
cd web
npm install --legacy-peer-deps
npm run dev -- --host 127.0.0.1 --port 5173
```

- UI：`http://127.0.0.1:5173`
- Vite 将 `/dashboard`、`/annotation`、`/benchmark`、`/prompt`、`/api` 代理到 `:8080`
- 侧栏填写 API Key=`dev-key`、Tenant=`tenant-dev`

页面：链路列表/详情、标注队列/详情、A/B 实验、统计分析、基准评测、运维健康。

### 3. 写入一条样例

```bash
curl -s http://127.0.0.1:8080/api/v1/ingest \
  -H 'X-API-Key: dev-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "spans": [{
      "trace_id": "aa11bb22cc33dd44ee55ff6677889900",
      "span_id": "aa11bb22cc33dd44",
      "name": "对话·好样本",
      "timestamp": "2026-07-17T06:00:00+00:00",
      "status": "OK",
      "kind": "LLM",
      "prompt": "一句话介绍杭州。",
      "response": "杭州是浙江省省会，以西湖风光与数字经济闻名。",
      "tags": ["好样本"]
    }]
  }'
```

坏样本可加 `"tags": ["safety"]` 并写入违规回复，以走安全 Judge → 归因（主链路）→ 标注（旁路）。  
全链路 live 脚本：`python scripts/live_e2e_deepseek.py`（需配置 Judge Key）。

## Docker Compose

```bash
cp .env.example .env   # 填密钥

# 单容器 DevStack + 可选前端
docker compose --profile monolith up -d --build
# API :8080  ·  UI :3000（若启用 web 服务）

# 多进程：gateway + ingest/dispatch/eval/attribution/annotation + Postgres + Redis Streams
docker compose --profile split up -d --build
```

## 数据库迁移

```bash
pip install -e ".[postgres,dev]"
# .env: LLMJUDGE_PERSISTENCE=sql
#       LLMJUDGE_DATABASE_URL=postgresql+asyncpg://llmj:llmj@127.0.0.1:5432/llmjudge
docker compose up -d postgres
llmj-db upgrade
```

`LLMJUDGE_SCHEMA_MODE`：`alembic`（默认 Postgres）/ `create_all`（SQLite 常见）/ `none`。

## CLI 入口

| 命令 | 作用 |
|------|------|
| `llmj-stack` | 单进程全链路（推荐本地） |
| `llmj-worker` | 按角色跑单个 worker（split） |
| `llmj-gateway` | 仅 Gateway |
| `llmj-ingest` / `llmj-dispatch` / `llmj-eval` | 各阶段 worker |
| `llmj-attribution` / `llmj-annotation` | 归因 / 标注 worker |
| `llmj-annotation-api` / `llmj-benchmark-api` | 独立 HTTP 服务 |
| `llmj-db` | Alembic 迁移 |

## 包结构

```text
src/llmobserverjudge/
  contracts/          # 契约模型与 Topic
  sdk/                # @observe + 异步上报
  gateway/            # FastAPI 摄取与鉴权
  ingest/             # 采样 / 索引 / 写入 / Outbox
  dispatch/           # eval_jobs 唯一生产方 + 后置路由
  eval_worker/        # Judge DAG、打分、执行图
  attribution_worker/ # 坏样本 ReAct 归因（主链路）
  annotation_worker/  # 人工标注存储与 API（旁路）
  prompt_admin/       # Prompt 版本与 A/B API
  dashboard/          # Trace / 统计 / WebSocket BFF
  benchmark/          # 批次评测
  runtime/            # DevStack / 多进程编排
  shared/             # MQ、持久化、环境变量、迁移
web/                  # React + TypeScript Dashboard
configs/              # dispatch.yaml、默认 Prompt
scripts/              # live E2E 等
tests/                # pytest
```

## SDK 示例

```python
from llmobserverjudge.sdk import configure, observe, flush, shutdown

configure(api_key="dev-key", endpoint="http://127.0.0.1:8080")

@observe("chat")
async def chat(q: str) -> str:
    return "你好，这是一条演示回复。"

# await chat("hi"); await flush(); await shutdown()
```

## HTTP Judge

设置 `LLMJUDGE_JUDGE_BASE_URL` + `LLMJUDGE_JUDGE_API_KEY` 后，DevStack / EvalWorker 会注册 OpenAI 兼容 HTTP Judge。未配置时使用内置启发式/规则 Judge（便于无 Key 联调）。

路由默认评估会绑定 Prompt `prompt_default_judge`；存在 RUNNING 的 A/B 实验时按 `trace_id` 哈希分流 control / treatment。

## 架构约束

- **Dispatch** 是 `eval_jobs` 的唯一生产方。
- **Eval Worker** 不得生产 `eval_jobs`；后置归因/标注经 `DispatchClient`。
- 执行图状态：归因 = 主链路（`path=main`）；标注 = 旁路（`path=side`）；看板时间线与执行图同源。

## 许可证

Proprietary
