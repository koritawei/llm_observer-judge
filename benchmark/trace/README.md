# trace_resolver — Langfuse 按需追踪模块

## 模块概述

`trace_resolver` 按业务 `trace_id` + `project_id` 定位 Langfuse 主键，并提供 **按需** observation 访问（skeleton / detail / find）。平台**不预拉**整包 `raw_trace`；业务侧通过 `TraceToolProvider`（provider 名 `trace`）调用 Observations API **v2**（`GET /api/public/v2/observations`）。v4 `events_only` 不提供旧的 `/api/public/traces` 与 `/api/public/observations`。

核心设计要点：

- **入口一次转换**：`EvaluatorService.before_eval` 用 metadata filter **+ 有界 timestamp lookback**（默认 14 天）把业务 id → Langfuse 主键；下游（tool / DB）只认主键。无时间窗的 list 在大项目上会 422/超时，且不得伪装成 `trace_not_found`。
- **单实例查询**：`project_id` → 唯一 Langfuse 实例。
- **Skeleton 投影**：list observations 的 input/output 压成 preview，禁止把 list 全文塞进 tool 结果。
- **Observation 翻页**：v2 用 `meta.cursor`，单页 100 条；`list_observations_for_trace` 按 cursor 拉取，对外仍返回扁平 list。`status` 映射为服务端 `level` 过滤。未请求的字段组不会出现在响应里。`input` / `output` 是原始字符串，客户端解析成对象后再交给工具。`metadata` 超过 200 字符时用 `expandMetadata` 再取一次全文。
- **失败语义**：未知 `project_id` → `None` / tool error；**list traces / observations / detail HTTP 失败**抛
  ``LangfuseHttpError``（list observations 为子类 ``LangfuseObservationsError``；入口 resolve 失败 → `langfuse_unavailable` 502；tool 层转 `from_error`），
  与「成功但 spans=[]」或 404 miss / 真正 `trace_not_found` 严格区分。Skeleton **不**调用重接口 `GET /traces/{id}`。
- **四个 tool 与入口不同**：skeleton / detail / find / full_trace 都按 **primary id 或 observation id** 查 v2。入口 metadata 反查仍用 14 天窗口；已知主键的点查用 10 年窗口，避免历史 trace 被短窗口判 miss。v2 每次请求仍带 `fromStartTime` / `toStartTime`。
- **sidecar observations**：`bind_sidecar_observations`（或兼容读取 `metadata.sidecar_observations`）有值时，四个读工具经 `sidecar_ingest.observations` 从这份 snake_case 树裁成同一套 ToolResult（与入口 TRACE 级 IO 抽取同模块），不打 Langfuse；缺 observation 直接报错，不回退 HTTP。sidecar 路径的 `fetch_full_trace` 不受 `enable_fetch_full_trace` 开关限制。

## 目录结构

```
trace_resolver/
├── __init__.py
├── schema.py            # Skeleton / ObservationDetail / Instance 模型
├── langfuse_client.py   # v2 HTTP：resolve_primary_id / list / get_observation
├── resolver.py          # TraceResolver 门面
└── tools.py             # TraceToolProvider + TraceRequestBound
```

## 公共 API

| 符号 | 说明 |
|---|---|
| `TraceResolver.resolve_primary_id_for` | **仅入口**：业务 id → 主键（metadata filter） |
| `TraceResolver.fetch_skeleton` | 主键 → span 骨架（preview） |
| `TraceResolver.fetch_observation` | `observation_id` → 全文 detail |
| `TraceResolver.find_observations` | 同 primary id 内按 name/type/status 过滤 |
| `TraceToolProvider` | agentgraph ToolProvider；`bind_trace_request` / `bound=` / `ToolExecutionContext.extra["bound"]` |
| `bind_trace_request` | 请求级 ContextVar，使 `ToolManager.call_tool(name, args)` 无需额外 kwargs |
| `bind_sidecar_observations` | evaluate() 期间挂 sidecar observation 树；`create_bound` 会拷到 bound |

| Tools | `fetch_trace_skeleton` / `fetch_observation_detail` / `find_observations` / `fetch_full_trace`（**慎用 / opt-in**，默认不注册） |

## 官方 API（自建实例）

| 用途 | 接口 |
|---|---|
| 入口：业务 id → 主键 | `GET /api/public/v2/observations?filter=[metadata.<key>=业务 id]&fromStartTime=&toStartTime=&limit=1&fields=core`（默认 lookback 14 天；主键取返回行的 `traceId`） |
| Skeleton / Find | `GET /api/public/v2/observations?traceId=&fields=core,basic,io&limit=100&cursor=`（已知主键，lookback 10 年） |
| Detail | 同路径，`filter` 为 `id = observationId`，`fields=core,basic,io,metadata,time`；截断的 metadata 再带 `expandMetadata` 请求一次 |
| 指标 | trace：`fields=core,io,metadata,metrics`，`expandMetadata=first_text_response_time_ms`。GENERATION：`fields=core,basic,io,usage,metrics`（token 读 `inputUsage` / `outputUsage` / `totalUsage`） |
| 全量 trace（**慎用**，默认不注册） | 同路径拉最多 1000 行，空 `fields` 请求宽字段组并附在 `observations` 上 |

自建 v4 `events_only` 与 Cloud 都走这条 v2。旧的 `GET /api/public/traces` 与 `GET /api/public/observations` 会 404。

## Bootstrap

YAML（框架级，业务 agent 不用再声明）::

```yaml
tools:
  providers:
    - trace
  extensions:
    trace:
      max_observation_chars: 65536  # 仅 skeleton / detail / find；不作用于 fetch_full_trace
      enable_fetch_full_trace: true  # 慎用；框架 conf 默认不打开
      timeout_seconds: 30            # Langfuse httpx 超时（默认 30）
      max_retries: 3                 # 仅 ConnectError / 429 / 5xx；超时不重试
      resolve_lookback_days: 14      # 入口 resolve 时间窗（防无界 list → 422）
```

> `max_observation_chars` **不**作用于 `fetch_full_trace`（全量 body 原样返回，不截断）。
> `enable_fetch_full_trace` 默认 `false`；框架 `conf/app.*.yaml` **不**写入 `true`。
> 大 trace 易导致 Langfuse 500、OOM 或撑爆 LLM context；推荐路径仍是 skeleton + detail。
> `timeout_seconds` / `max_retries` / `resolve_lookback_days` 经 `TraceToolExt` → `TraceResolver`；超时（ReadTimeout）**不**重试。
最小 bootstrap（仅注册默认 3 个 tool）::

```yaml
tools:
  providers:
    - trace
```

装配链（缺一不可）:

1. **`tools.py` import 副作用**：`ProviderRegistry.register_metadata_adapter("tool", _build_tool_component_provider)`。  
   `agentgraph.tools` 不像 `agents` 那样 eager 注册 tool adapter；agent-eval 也不依赖 `libs/tools`（那边 import 才会注册）。不补这一步会出现：
   `failed to register ...TraceToolProvider` → `LookupError: provider not found: kind='tool' name='trace'`。
2. **`@tool_component(name="trace")`**：让 discover 把类登记进 `ProviderRegistry`。
3. **YAML `tools.providers: [trace]`**：bootstrap `_install_component("tool")` 调 `from_config`。
4. **`from_config`**：读 `AGENT_EVAL_LANGFUSE_INSTANCES` 建 `TraceResolver`；env 为空则返回 `None`（跳过实例，但 registry 里仍应有 `tool/trace` ComponentProvider）。

### 凭据切开（读 vs 写）

| 用途 | 环境变量 | 说明 |
|---|---|---|
| **拉数**（skeleton / detail / find / full_trace HTTP） | `AGENT_EVAL_LANGFUSE_INSTANCES` | 按被测 `project_id` 选实例；只读 |
| **上报**（`call_tool` 内 `trace.*` SPAN） | `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`（经 `get_report_client()`，显式 `public_key`） | 与主链路 `LangfuseMiddleware` **同一客户端**；`start_observation` + `with_current_node_otel_parent()` 挂到当前 LangGraph node（或 triage 容器），**禁止**新 `Langfuse()` 另起 root |

缺父 trace / 上报 key 时 SPAN 跳过；拉数仍走 INSTANCES，互不影响。
SPAN output 会记录具体工具结果：合法 JSON 写为结构化 `data`，其它内容写为文本；
超过 `max_observation_chars` 时截断并设置 `body_truncated=true`。

`app/main.py` 须 `import TraceToolProvider`（触发 1+2），再 `runtime.bootstrap()`。

业务侧::

```python
provider = runtime.tool_manager.get_provider("trace")
bound = provider.bound_from_metadata(metadata)
await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)
```
