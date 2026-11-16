#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# trigger_pipeline_benchmark.sh — Pipeline Benchmark (Kafka 历史评分回溯) 触发脚本
#
# 流程对齐 BenchmarkRunner.run_pipeline_benchmark：
#   sample_kafka_pipeline_records → load_kafka_criterion_entries →
#   AggregationCalculator → complete_run(metrics_json)
#
# 与常规 benchmark 不同：不调 LLM、不拉 Langfuse dataset、不写合成 trace_id 行。
# 仅 SQL 聚合 Kafka 来源 trace_pipeline_records 的已有 judge_json.score。
#
# 仅异步：POST 立即返回 202 + run_id，脚本默认直接退出；任务在 agent-eval
# 进程内继续执行。可选 POLL=true 轮询终态。
# 服务端进程内串行：与常规 benchmark 共享 _job_lock，同一时间只跑一个。
#
# 用法:
#   BENCHMARK_NAME="example_benchmark" \
#   DATE_START="2026-08-04" \
#   DATE_END="2026-08-05" \
#   bash scripts/trigger_pipeline_benchmark.sh
#
#   # 提交后默认立即退出（fire-and-forget）；任务在服务端继续跑。
#   # 仅当需要阻塞等待终态时再设 POLL=true。
#
#   # 仅列出已注册 benchmark
#   LIST_ONLY=true bash scripts/trigger_pipeline_benchmark.sh
#
# 可选环境变量:
#   AGENT_EVAL_URL       — agent-eval 服务地址 (默认 https://pre-aiagent-eval.amap-inc.com)
#   BENCHMARK_NAME       — @benchmark_component name（必填，除非 LIST_ONLY）
#   DATE_START           — 起始日 yyyy-mm-dd（+08 当日 00:00:00）；也接受完整带时区时间
#   DATE_END             — 结束日 yyyy-mm-dd（+08 当日 23:59:59.999999）；也接受完整带时区时间
#   POLL                 — true 时提交后轮询终态（默认 false：提交成功即退出）
#   SAMPLE_SIZE          — 抽样条数；空 = 全量扫描（取最新 N 条，非随机）
#   SAMPLE_SEED          — 采样种子 (默认 42，当前保留字段)
#   GENERATE_REPORT      — true|false (默认 true)
#   PROJECT_ID           — 按 Langfuse project_id 过滤；空 = 全部项目
#   CURL_TIMEOUT         — 仅 POLL=true 时轮询总超时秒数 (默认 600)
#   POLL_INTERVAL        — 轮询间隔秒数 (默认 3)
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

# ── 参数 ──────────────────────────────────────────────
AGENT_EVAL_URL="${AGENT_EVAL_URL:-https://pre-aiagent-eval.amap-inc.com}"
LIST_ONLY="${LIST_ONLY:-false}"
BENCHMARK_NAME="${BENCHMARK_NAME:-}"
DATE_START="${DATE_START:-}"
DATE_END="${DATE_END:-}"
POLL="${POLL:-false}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
GENERATE_REPORT="${GENERATE_REPORT:-true}"
PROJECT_ID="${PROJECT_ID:-}"
CURL_TIMEOUT="${CURL_TIMEOUT:-600}"
POLL_INTERVAL="${POLL_INTERVAL:-3}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-10}"
KEEPALIVE_TIME="${KEEPALIVE_TIME:-30}"

# ── 校验 ──────────────────────────────────────────────
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

if [ "${LIST_ONLY}" != "true" ]; then
  if [ -z "${BENCHMARK_NAME}" ]; then
    error "BENCHMARK_NAME is required (e.g. example_benchmark), or set LIST_ONLY=true"
    exit 1
  fi
  if ! [[ "${BENCHMARK_NAME}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    error "BENCHMARK_NAME contains invalid characters (allowed: a-z A-Z 0-9 _ -): ${BENCHMARK_NAME}"
    exit 1
  fi
  if [ -z "${DATE_START}" ] || [ -z "${DATE_END}" ]; then
    error "DATE_START and DATE_END are required (yyyy-mm-dd, e.g. 2026-08-04)"
    exit 1
  fi
fi

# ── 日期规范化：yyyy-mm-dd → Asia/Shanghai(+08) 当日边界 ──
# DATE_START=yyyy-mm-dd → {date} 00:00:00.000000+08:00
# DATE_END=yyyy-mm-dd   → {date} 23:59:59.999999+08:00（配合 SQL <=）
# 完整时间串：有时区则换算到 +08；无时区按 +08 解释。API 需 +08:00（非 PG 显示的 +08）
normalize_pipeline_dates() {
  python3 -c '
import re, sys
from datetime import datetime
from zoneinfo import ZoneInfo

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TZ_CN = ZoneInfo("Asia/Shanghai")

def fmt_cn(dt: datetime) -> str:
    local = dt.astimezone(TZ_CN)
    # 与 DB 展示对齐：空格分隔 + 微秒；API 要求 +08:00（裸 +08 pydantic 不认）
    return local.strftime("%Y-%m-%d %H:%M:%S.%f") + "+08:00"

def norm(value: str, *, end_of_day: bool) -> str:
    raw = value.strip()
    if DATE_RE.fullmatch(raw):
        y, m, d = map(int, raw.split("-"))
        if end_of_day:
            dt = datetime(y, m, d, 23, 59, 59, 999999, tzinfo=TZ_CN)
        else:
            dt = datetime(y, m, d, 0, 0, 0, 0, tzinfo=TZ_CN)
        return fmt_cn(dt)
    # Full datetime: Z / offset / naive(+08)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(
            f"invalid date {value!r}: use yyyy-mm-dd or ISO8601 ({exc})"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_CN)
    return fmt_cn(dt)

start = norm(sys.argv[1], end_of_day=False)
end = norm(sys.argv[2], end_of_day=True)
if end <= start:
    raise SystemExit(
        f"DATE_END must be after DATE_START after normalize: {start} .. {end}"
    )
print(start)
print(end)
' "$1" "$2"
}

if [ "${LIST_ONLY}" != "true" ]; then
  _NORM_OUT=$(normalize_pipeline_dates "${DATE_START}" "${DATE_END}") || {
    error "${_NORM_OUT:-date normalize failed}"
    exit 1
  }
  DATE_START_RAW="${DATE_START}"
  DATE_END_RAW="${DATE_END}"
  DATE_START=$(printf '%s\n' "${_NORM_OUT}" | sed -n '1p')
  DATE_END=$(printf '%s\n' "${_NORM_OUT}" | sed -n '2p')
  unset _NORM_OUT
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

# ── 构建 JSON body（对齐 PipelineBenchmarkRequest）──
info "Building request payload..."
BODY=$(python3 -c "
import json, sys
body = {
    'date_start': sys.argv[1],
    'date_end': sys.argv[2],
    'sample_seed': int(sys.argv[3]),
    'generate_report': sys.argv[4].lower() == 'true',
}
if sys.argv[5]:
    body['sample_size'] = int(sys.argv[5])
if sys.argv[6]:
    body['project_id'] = sys.argv[6]
print(json.dumps(body))
" "$DATE_START" "$DATE_END" "$SAMPLE_SEED" "$GENERATE_REPORT" \
  "$SAMPLE_SIZE" "$PROJECT_ID")

# ── 打印请求信息 ─────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Pipeline Benchmark — Kafka Retrospective Aggregation"
echo "═══════════════════════════════════════════════════════════"
echo "  Service URL      : ${AGENT_EVAL_URL}"
echo "  Benchmark        : ${BENCHMARK_NAME}"
echo "  Mode             : $([ "${POLL}" = "true" ] && echo "submit+poll" || echo "fire-and-forget (exit after 202)")"
echo "  Date input       : ${DATE_START_RAW:-${DATE_START}} .. ${DATE_END_RAW:-${DATE_END}}"
echo "  Date range (+08) : ${DATE_START} .. ${DATE_END}"
echo "  Sample size      : ${SAMPLE_SIZE:-(full scan)}"
echo "  Sample seed      : ${SAMPLE_SEED}"
echo "  Generate report  : ${GENERATE_REPORT}"
echo "  Project ID       : ${PROJECT_ID:-(all projects)}"
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
info "Submitting pipeline benchmark run..."
CURL_EXIT=0
CURL_ERR=$(mktemp)
SUBMIT_MAX_TIME=30

HTTP_CODE=$(curl -sS -o "$TMP_RESP" -w "%{http_code}" \
  -X POST \
  "${AGENT_EVAL_URL}/v1/benchmarks/${BENCHMARK_NAME}/run-pipeline" \
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

RUN_ID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('run_id',''))" "$TMP_RESP" 2>/dev/null) || RUN_ID=""
POLL_URL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('poll_url',''))" "$TMP_RESP" 2>/dev/null) || POLL_URL=""
if [ -z "${RUN_ID}" ]; then
  error "Failed to extract run_id from response:"
  cat "$TMP_RESP"
  exit 1
fi

ok "Run accepted: run_id=${RUN_ID}"
info "Status URL: ${AGENT_EVAL_URL}${POLL_URL:-/v1/benchmarks/runs/${RUN_ID}}"
info "Task runs on agent-eval server; script does not block by default."

if [ "${POLL}" != "true" ]; then
  echo ""
  ok "Submitted — exiting now (server continues)."
  info "Check status later:"
  echo "  curl -sS ${AGENT_EVAL_URL}/v1/benchmarks/runs/${RUN_ID}"
  exit 0
fi

warn "POLL=true — blocking until terminal status (interval=${POLL_INTERVAL}s, timeout=${CURL_TIMEOUT}s)..."
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

  STATUS=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('status','unknown'))" "$TMP_RESP" 2>/dev/null) || STATUS=""
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
" "$TMP_RESP" 2>/dev/null) || DONE="?/?"
  TASKS=$(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
t=(d.get('progress') or {}).get('tasks') or {}
if t:
    print('tasks=%s/%s/%s/%s' % (t.get('completed',0),t.get('running',0),t.get('pending',0),t.get('failed',0)))
" "$TMP_RESP" 2>/dev/null) || TASKS=""

  case " ${TERMINAL_STATUSES} " in
    *" ${STATUS} "*)
      if [ "$STATUS" = "completed" ]; then
        ok "Pipeline benchmark completed (elapsed=${ELAPSED}s)"
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
print("  NOTE: No synthetic trace_id rows (data from Kafka daily records)")
print(f"  Metrics:    GET /v1/benchmarks/runs/{d.get('run_id')}/metrics")
PYEOF
        break
      else
        error "Pipeline benchmark failed (elapsed=${ELAPSED}s):"
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
