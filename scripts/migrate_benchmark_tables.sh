#!/usr/bin/env bash
# 运维包装：把 bullMQ benchmark 表数据迁入 main 表结构（psycopg3）。
#
# 源: benchmark_runs_bullmq, benchmark_run_items_bullmq
# 目标: benchmark_runs, benchmark_run_items
# 已存在的 run_id / (run_id, item_id, criterion_id) / id 自动跳过，只插入 diff。
#
# 用法（默认 dry-run；加 --execute 才写目标库）:
#   bash scripts/migrate_benchmark_tables.sh \
#     --source-url 'postgresql://USER:PASSWORD@bullmq-host:5432/agent_eval' \
#     --target-url 'postgresql://USER:PASSWORD@main-host:5432/agent_eval'
#   bash scripts/migrate_benchmark_tables.sh \
#     --source-url 'postgresql://USER:PASSWORD@bullmq-host:5432/agent_eval' \
#     --target-url 'postgresql://USER:PASSWORD@main-host:5432/agent_eval' \
#     --execute
#
# --source-url 指向 bullMQ 库，--target-url 指向 main 库。线上会 source setenv.sh。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export HOME="${HOME:-/home/admin}"
export APP_HOME="${APP_HOME:-${HOME}/agent-eval-service}"
export ADMIN_HOME="${ADMIN_HOME:-${HOME}}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PATH="${PATH:-/usr/local/bin:/usr/bin:/bin}"

_SETENV="${APP_HOME}/base/bin/setenv.sh"
if [[ -f "${_SETENV}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${_SETENV}" || true
  set -u
fi

REPO_ROOT=""
if [[ -d "${ROOT}/../../libs/agent_graph/src/agentgraph" ]]; then
  REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
fi

SCRIPT_PY="${ROOT}/scripts/migrate_benchmark_tables.py"
if [[ ! -f "${SCRIPT_PY}" ]]; then
  echo "[migrate_benchmark_tables] FATAL: migrate_benchmark_tables.py not found" >&2
  exit 1
fi

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
  elif [[ -n "${REPO_ROOT}" && -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

echo "[migrate_benchmark_tables] python=${PYTHON}" >&2
echo "[migrate_benchmark_tables] ROOT=${ROOT}" >&2

if ! "${PYTHON}" -c "import psycopg" 2>/dev/null; then
  echo "[migrate_benchmark_tables] FATAL: ${PYTHON} cannot import psycopg (need psycopg 3)" >&2
  exit 1
fi

exec "${PYTHON}" "${SCRIPT_PY}" "$@"
