#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# trigger_benchmark.sh — Benchmark 离线评测触发脚本
#
# 流程对齐 BenchmarkRunner：
#   create_run(job id) → 拉 Langfuse dataset → 逐 criterion 评测落 pipeline
#   → SQL 聚合 → AggregationCalculator → complete_run(metrics_json)
#
# 仅异步：POST 立即返回 202 + run_id，脚本默认直接退出；任务在 agent-eval
# 进程内继续执行。可选 POLL=true 轮询终态。
# 服务端进程内串行：多 POST 均 202，后台同一时间只跑一个 benchmark。
#
# 用法:
#   BENCHMARK_NAME="example_benchmark" \
#   bash scripts/trigger_benchmark.sh
#
#   # 提交后轮询直到完成
#   POLL=true bash scripts/trigger_benchmark.sh
#
#   # 仅列出已注册 benchmark
#   LIST_ONLY=true bash scripts/trigger_benchmark.sh
#
# 可选环境变量:
#   AGENT_EVAL_URL       — agent-eval 服务地址 (默认 https://pre-aiagent-eval.amap-inc.com)
#   BENCHMARK_NAME       — @benchmark_component name（必填，除非 LIST_ONLY）
#   POLL                 — true 时提交后轮询终态 (默认 false，提交即退出)
#   SAMPLE_SIZE          — 采样条数；空 = 用 benchmark 默认（全量或 YAML sample_size）
#   SAMPLE_SEED          — 采样种子 (默认 42)
#   MAX_CONCURRENCY      — 固定 readonly 1（禁止外部传入，避免多 worker 并发）
#   PARALLEL_EVALUATORS  — true|false；空 = 服务 YAML 默认
#   TRIAGE_MODE          — fused | per_eval | none (默认 fused)
#   DATASET_NAME         — 覆盖 Langfuse dataset name（空 = 用 benchmark 默认）
#   PROJECT_ID          — 覆盖 Langfuse project ID（空 = 用 benchmark 默认）
#   GENERATE_REPORT      — true|false (默认 true)
#   RUBRIC_REF           — Rubric 类引用 module:Class（空 = benchmark 默认，§5.1）
#   PARSE_INTERCEPTOR_REF — 数据 parse 拦截器引用 module:Class（空 = Rubric 类挂载值）
#   JUDGE_TARGET         — judge target（async=完整 trigger URL / sync=本地类名；空=默认）
#   ATTR_TARGET          — 归因 target（语义同 JUDGE_TARGET；空=默认）
#   CURL_TIMEOUT         — 仅 POLL=true 时轮询总超时秒数 (默认 3600)
#   POLL_INTERVAL        — 轮询间隔秒数 (默认 5)
#   CONNECT_TIMEOUT      — curl 连接超时秒数 (默认 10)
#   KEEPALIVE_TIME       — TCP keepalive 间隔秒数 (默认 30)
#   DEBUG                — true 时打印请求 body
# ═══════════════════════════════════════════════════════════════════════════

# ── 颜色输出 ──────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# 从 $TMP_RESP 提取 JSON 字段；解析失败 warn 响应片段（不静默吞错）
json_get() {
  local key="$1"
  python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" \
    "$TMP_RESP" "$key" 2>/dev/null || {
    warn "JSON parse failed for '${key}': $(head -c 200 "$TMP_RESP" 2>/dev/null || echo '<unreadable>')"
    return 1
  }
}

# ── 参数 ──────────────────────────────────────────────
AGENT_EVAL_URL="${AGENT_EVAL_URL:-https://pre-aiagent-eval.amap-inc.com}"
LIST_ONLY="${LIST_ONLY:-false}"
BENCHMARK_NAME="${BENCHMARK_NAME:-}"
POLL="${POLL:-false}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
# 并发数固定为 1：readonly 禁止外部环境变量/后续代码覆盖（避免多 worker 并发）
readonly MAX_CONCURRENCY=1
PARALLEL_EVALUATORS="${PARALLEL_EVALUATORS:-}"
TRIAGE_MODE="${TRIAGE_MODE:-fused}"
GENERATE_REPORT="${GENERATE_REPORT:-true}"
DATASET_NAME="${DATASET_NAME:-}"
PROJECT_ID="${PROJECT_ID:-}"
# 配置化运行时（§5.1）：可选 profile 字段，空 = 现网行为不变
RUBRIC_REF="${RUBRIC_REF:-}"
PARSE_INTERCEPTOR_REF="${PARSE_INTERCEPTOR_REF:-}"
JUDGE_TARGET="${JUDGE_TARGET:-}"
ATTR_TARGET="${ATTR_TARGET:-}"
CURL_TIMEOUT="${CURL_TIMEOUT:-3600}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-10}"
KEEPALIVE_TIME="${KEEPALIVE_TIME:-30}"

# ── 校验 ──────────────────────────────────────────────
case "${TRIAGE_MODE}" in
  fused|per_eval|none) ;;
  *)
    error "TRIAGE_MODE must be fused|per_eval|none, got: ${TRIAGE_MODE}"
    exit 1
    ;;
esac

if [ -n "${PARALLEL_EVALUATORS}" ]; then
  case "${PARALLEL_EVALUATORS}" in
    true|false) ;;
    *)
      error "PARALLEL_EVALUATORS must be true|false, got: ${PARALLEL_EVALUATORS}"
      exit 1
      ;;
  esac
fi

for _var in SAMPLE_SEED CURL_TIMEOUT POLL_INTERVAL CONNECT_TIMEOUT KEEPALIVE_TIME; do
  _val="${!_var}"
  if ! [[ "${_val}" =~ ^[1-9][0-9]*$ ]] && ! [[ "${_val}" =~ ^0$ ]]; then
    error "${_var} must be a non-negative integer, got: ${_val}"
    exit 1
  fi
done
unset _var _val

if [ -n "${SAMPLE_SIZE}" ] && ! [[ "${SAMPLE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  error "SAMPLE_SIZE must be a positive integer, got: ${SAMPLE_SIZE}"
  exit 1
fi

if [ "${LIST_ONLY}" != "true" ] && [ -z "${BENCHMARK_NAME}" ]; then
  error "BENCHMARK_NAME is required (e.g. example_benchmark), or set LIST_ONLY=true"
  exit 1
fi

if [ "${LIST_ONLY}" != "true" ] && ! [[ "${BENCHMARK_NAME}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  error "BENCHMARK_NAME contains invalid characters (allowed: a-z A-Z 0-9 _ -): ${BENCHMARK_NAME}"
  exit 1
fi

if [ "${POLL}" = "true" ] && [ "${CURL_TIMEOUT}" -lt 300 ]; then
  warn "CURL_TIMEOUT=${CURL_TIMEOUT}s is low for benchmark polling"
fi

# ── 临时文件 ──────────────────────────────────────────
TMP_RESP=$(mktemp)
trap 'rm -f "$TMP_RESP" "${CURL_ERR:-}"' EXIT INT TERM

# ── 健康检查 ──────────────────────────────────────────
info "Checking service health..."
HEALTH_RESP=$(curl -sS -o /dev/null -w "%{http_code}" \
  --max-time 5 \
  "${AGENT_EVAL_URL}/health" 2>/dev/null || true)

if [ "${HEALTH_RESP}" != "200" ]; then
  warn "Health check returned: ${HEALTH_RESP:-unreachable}"
  warn "Proceeding anyway..."
else
  ok "Service is healthy"
fi
echo ""

# ── 列出已注册 benchmark ──────────────────────────────
info "Fetching registered benchmarks..."
CURL_EXIT=0
HTTP_CODE=$(curl -sS -o "$TMP_RESP" -w "%{http_code}" \
  "${AGENT_EVAL_URL}/v1/benchmarks" \
  --connect-timeout "${CONNECT_TIMEOUT}" \
  --max-time 15 2>/dev/null) || CURL_EXIT=$?

if [ "$CURL_EXIT" -eq 0 ] && [ "${HTTP_CODE}" = "200" ]; then
  python3 - "$TMP_RESP" <<'PYEOF'
import json, sys
rows = json.load(open(sys.argv[1]))
if not rows:
    print("  (no benchmarks registered)")
else:
    print(f"  {'name':28s}  {'service':16s}  dataset")
    print(f"  {'-'*28}  {'-'*16}  {'-'*24}")
    for r in rows:
        print(
            f"  {r.get('name',''):28s}  "
            f"{r.get('service_name',''):16s}  "
            f"{r.get('dataset_name','')}"
        )
print()
PYEOF
else
  warn "Could not list benchmarks (HTTP ${HTTP_CODE:-n/a})"
fi

if [ "${LIST_ONLY}" = "true" ]; then
  info "LIST_ONLY=true — exit without triggering a run."
  exit 0
fi

# ── 构建 JSON body（snake_case，对齐 BenchmarkRunRequest）──
info "Building request payload..."
BODY=$(python3 -c "
import json, sys
body = {
    'sample_seed': int(sys.argv[1]),
    'triage_mode': sys.argv[2],
    'generate_report': sys.argv[3].lower() == 'true',
}
if sys.argv[4]:
    body['parallel_evaluators'] = sys.argv[4].lower() == 'true'
if sys.argv[5]:
    body['sample_size'] = int(sys.argv[5])
if sys.argv[6]:
    body['max_concurrency'] = int(sys.argv[6])
if sys.argv[7]:
    body['dataset_name'] = sys.argv[7]
if sys.argv[8]:
    body['project_id'] = sys.argv[8]
# 配置化运行时（§5.1 可选 profile 字段）
if sys.argv[9]:
    body['rubric_ref'] = sys.argv[9]
if sys.argv[10]:
    body['parse_interceptor_ref'] = sys.argv[10]
if sys.argv[11]:
    body['judge'] = {'target': sys.argv[11]}
if sys.argv[12]:
    body['attribution'] = {'target': sys.argv[12]}
print(json.dumps(body))
" "$SAMPLE_SEED" "$TRIAGE_MODE" "$GENERATE_REPORT" \
  "$PARALLEL_EVALUATORS" "$SAMPLE_SIZE" "$MAX_CONCURRENCY" \
  "$DATASET_NAME" "$PROJECT_ID" \
  "$RUBRIC_REF" "$PARSE_INTERCEPTOR_REF" "$JUDGE_TARGET" "$ATTR_TARGET")

# ── 打印请求信息 ─────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Benchmark Offline Evaluation (async fire-and-forget)"
echo "═══════════════════════════════════════════════════════════"
echo "  Service URL      : ${AGENT_EVAL_URL}"
echo "  Benchmark        : ${BENCHMARK_NAME}"
echo "  Poll after submit: ${POLL}"
echo "  Sample size      : ${SAMPLE_SIZE:-(benchmark default)}"
echo "  Sample seed      : ${SAMPLE_SEED}"
echo "  Max concurrency  : ${MAX_CONCURRENCY} (fixed)"
echo "  Parallel evals   : ${PARALLEL_EVALUATORS:-(YAML default)}"
echo "  Triage mode      : ${TRIAGE_MODE}"
echo "  Generate report  : ${GENERATE_REPORT}"
echo "  Dataset name    : ${DATASET_NAME:-(benchmark default)}"
echo "  Project ID      : ${PROJECT_ID:-(benchmark default)}"
echo "  Rubric ref      : ${RUBRIC_REF:-(benchmark default)}"
echo "  Parse intercept : ${PARSE_INTERCEPTOR_REF:-(rubric class default)}"
echo "  Judge target    : ${JUDGE_TARGET:-(process default)}"
echo "  Attr target     : ${ATTR_TARGET:-(process default)}"
if [ "${POLL}" = "true" ]; then
  echo "  Poll timeout     : ${CURL_TIMEOUT}s"
fi
echo "═══════════════════════════════════════════════════════════"
echo ""

if [ "${DEBUG:-false}" = "true" ]; then
  info "Request body:"
  echo "${BODY}" | python3 -m json.tool 2>/dev/null || echo "${BODY}"
  echo ""
fi

# ── 提交 run ──────────────────────────────────────────
info "Submitting benchmark run..."
CURL_EXIT=0
CURL_ERR=$(mktemp)
SUBMIT_MAX_TIME=30

HTTP_CODE=$(curl -sS -o "$TMP_RESP" -w "%{http_code}" \
  -X POST \
  "${AGENT_EVAL_URL}/v1/benchmarks/${BENCHMARK_NAME}/run" \
  -H "Content-Type: application/json" \
  --connect-timeout "${CONNECT_TIMEOUT}" \
  --keepalive-time "${KEEPALIVE_TIME}" \
  --max-time "${SUBMIT_MAX_TIME}" \
  -d "${BODY}" 2>"$CURL_ERR") || CURL_EXIT=$?

if [ "$CURL_EXIT" -ne 0 ]; then
  error "Failed to submit run (exit=$CURL_EXIT): $(cat "$CURL_ERR")"
  rm -f "$CURL_ERR"
  exit 1
fi
rm -f "$CURL_ERR"

if [ "${HTTP_CODE}" != "202" ]; then
  error "Submit failed with HTTP ${HTTP_CODE} (expected 202)"
  cat "$TMP_RESP" 2>/dev/null
  exit 1
fi

RUN_ID=$(json_get run_id) || RUN_ID=""
POLL_URL=$(json_get poll_url) || POLL_URL=""
if [ -z "${RUN_ID}" ]; then
  error "Failed to extract run_id from response:"
  cat "$TMP_RESP"
  exit 1
fi

ok "Run accepted: run_id=${RUN_ID}"
info "Status URL: ${AGENT_EVAL_URL}${POLL_URL:-/v1/benchmarks/runs/${RUN_ID}}"
info "agent-eval continues in background; client may exit now."

if [ "${POLL}" != "true" ]; then
  echo ""
  info "Done (fire-and-forget). Re-check later with:"
  echo "  curl -sS ${AGENT_EVAL_URL}/v1/benchmarks/runs/${RUN_ID}"
  exit 0
fi

info "POLL=true — polling (interval=${POLL_INTERVAL}s, timeout=${CURL_TIMEOUT}s)..."
echo ""

START_TIME=$(date +%s)
TERMINAL_STATUSES="completed failed"

while true; do
  ELAPSED=$(( $(date +%s) - START_TIME ))
  if [ "$ELAPSED" -ge "$CURL_TIMEOUT" ]; then
    error "Polling timed out after ${ELAPSED}s (CURL_TIMEOUT=${CURL_TIMEOUT})"
    error "Server may still be running. Check GET ${AGENT_EVAL_URL}/v1/benchmarks/runs/${RUN_ID}"
    exit 1
  fi

  CURL_EXIT=0
  HTTP_CODE=$(curl -sS -o "$TMP_RESP" -w "%{http_code}" \
    "${AGENT_EVAL_URL}/v1/benchmarks/runs/${RUN_ID}" \
    --connect-timeout "${CONNECT_TIMEOUT}" \
    --keepalive-time "${KEEPALIVE_TIME}" \
    --max-time 15 2>/dev/null) || CURL_EXIT=$?

  if [ "$CURL_EXIT" -ne 0 ]; then
    warn "Poll request failed (exit=$CURL_EXIT), retrying in ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
    continue
  fi

  if [ "${HTTP_CODE}" = "404" ]; then
    error "Run not found: ${RUN_ID}"
    exit 1
  fi

  if [ "${HTTP_CODE}" != "200" ]; then
    warn "Poll returned HTTP ${HTTP_CODE}, retrying in ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
    continue
  fi

  STATUS=$(json_get status) || STATUS="unknown"
  DONE=$(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
completed=d.get('completed_evaluators',0)
total_ev=d.get('total_evaluators',0)
# Run-row counters may lag in distributed mode; fall back to
# real-time task stats (from benchmark_tasks GROUP BY).
tasks=(d.get('progress') or {}).get('tasks') or {}
if completed==0 and tasks:
    completed=tasks.get('completed',0)
if total_ev==0 and tasks:
    total_ev=tasks.get('total',0)
print('%s/%s' % (completed,total_ev))
" "$TMP_RESP" 2>/dev/null) || {
    DONE="?/?"
    warn "JSON parse failed for progress: $(head -c 200 "$TMP_RESP" 2>/dev/null || echo '<unreadable>')"
  }
  TASKS=$(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
t=(d.get('progress') or {}).get('tasks') or {}
if t:
    print('tasks=%s/%s/%s/%s' % (t.get('completed',0),t.get('running',0),t.get('pending',0),t.get('failed',0)))
" "$TMP_RESP" 2>/dev/null) || {
    TASKS=""
    warn "JSON parse failed for tasks: $(head -c 200 "$TMP_RESP" 2>/dev/null || echo '<unreadable>')"
  }

  case " ${TERMINAL_STATUSES} " in
    *" ${STATUS} "*)
      if [ "$STATUS" = "completed" ]; then
        ok "Benchmark completed (elapsed=${ELAPSED}s)"
        echo ""
        python3 - "$TMP_RESP" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  run_id              : {d.get('run_id')}")
print(f"  benchmark_name      : {d.get('benchmark_name')}")
print(f"  service_name        : {d.get('service_name')}")
print(f"  dataset_name        : {d.get('dataset_name')}")
print(f"  status              : {d.get('status')}")
print(f"  final_score         : {d.get('final_score')}")
print(f"  total_samples       : {d.get('total_samples')}")
print(f"  evaluators          : {d.get('completed_evaluators')}/{d.get('total_evaluators')}")
print(f"  duration_ms         : {d.get('duration_ms')}")
report = (d.get('report_meta_json') or {}).get('report_url')
print(f"  report_url          : {report}")
metrics = d.get('metrics_json') or {}
cm = metrics.get('criterion_metrics') or {}
if cm:
    print("  criterion_metrics:")
    for cid, stats in cm.items():
        if not isinstance(stats, dict):
            continue
        print(
            f"    [{cid}] mean={stats.get('mean')} "
            f"median={stats.get('median')} p95={stats.get('p95')} "
            f"pass_rate={stats.get('pass_rate')} "
            f"success={stats.get('success_count')} "
            f"failed={stats.get('failed_count')}"
        )
attr = metrics.get('attribution_metrics') or {}
if attr:
    print(f"  attribution_metrics keys: {', '.join(sorted(attr.keys()))}")
print()
print("  Case details: trace_pipeline_records WHERE trace_id LIKE")
print(f"    'benchmark:{d.get('run_id')}:%'")
print(f"  Poll again: GET /v1/benchmarks/runs/{d.get('run_id')}")
print(f"  Metrics:    GET /v1/benchmarks/runs/{d.get('run_id')}/metrics")
PYEOF
        break
      else
        error "Benchmark failed (elapsed=${ELAPSED}s):"
        python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('error_message') or 'unknown')" "$TMP_RESP" 2>/dev/null || cat "$TMP_RESP"
        exit 1
      fi
      ;;
    *)
      info "Status: ${STATUS} progress=${DONE} ${TASKS} (elapsed=${ELAPSED}s)"
      sleep "$POLL_INTERVAL"
      ;;
  esac
done

echo ""
info "Done."
