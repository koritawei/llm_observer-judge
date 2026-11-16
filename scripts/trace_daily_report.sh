#!/usr/bin/env bash
# 运维包装：trace 日级报表（T+1）dry-run / 真实出报
#
# 链路同 TraceDailyReportScheduler._run_item（见 scripts/trace_daily_report_dryrun.py）。
#
# 用法:
#   bash scripts/trace_daily_report.sh
#   bash scripts/trace_daily_report.sh --name amap_agent_daily --date 20260928
#   bash scripts/trace_daily_report.sh --name amap_agent_daily --execute
#   bash scripts/trace_daily_report.sh --name amap_agent_daily --date 20260929 --execute --force
#   bash scripts/trace_daily_report.sh --config-json docs/remote_config-trace-daily-report-config-example.json
#
# 便捷环境变量（会转成对应 CLI 参数，显式传参优先）:
#   REPORT_NAME   → --name
#   DATA_DATE     → --date (yyyymmdd)
#   CONFIG_JSON   → --config-json
#   EXECUTE=true  → --execute
#   FORCE=true    → --force（须同时 EXECUTE=true；先删同名 run 再出报）
#
# 线上（APP_HOME=/home/admin/agent-eval-service）须与 bazel_docker_server.sh 一致，
# 把 libs/agent_graph 等拼进 PYTHONPATH；勿直接用裸 python3。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# ── 部署布局：APP_HOME vs Bazel workspace ───────────────────────────────
export HOME="${HOME:-/home/admin}"
BAZEL_WS="${BAZEL_WORKSPACE_HOME:-${HOME}/bazel-workspace}"

SERVICE_ROOT="${ROOT}"
if [[ -d "${BAZEL_WS}/services/agent-eval/app" ]]; then
  SERVICE_ROOT="${BAZEL_WS}/services/agent-eval"
elif [[ -d "${HOME}/bazel-workspace/services/agent-eval/app" ]]; then
  BAZEL_WS="${HOME}/bazel-workspace"
  SERVICE_ROOT="${BAZEL_WS}/services/agent-eval"
fi

REPO_ROOT=""
if [[ -d "${ROOT}/../../libs/agent_graph/src/agentgraph" ]]; then
  REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
fi

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

export PYTHONPATH="${SERVICE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for _usp in "${HOME}"/.local/lib/python3.*/site-packages; do
  [[ -d "${_usp}" ]] && export PYTHONPATH="${PYTHONPATH}:${_usp}"
done

_add_path_if_pkg() {
  local cand="$1" pkg="$2"
  if [[ -d "${cand}/${pkg}" ]]; then
    export PYTHONPATH="${cand}${PYTHONPATH:+:${PYTHONPATH}}"
    return 0
  fi
  return 1
}

agentgraph_ok=0
for cand in \
  "${BAZEL_WS}/libs/agent_graph/src" \
  "${REPO_ROOT:+${REPO_ROOT}/libs/agent_graph/src}" \
  "${HOME}/bazel-workspace/libs/agent_graph/src" \
  "/home/admin/bazel-workspace/libs/agent_graph/src"; do
  [[ -z "${cand}" ]] && continue
  if _add_path_if_pkg "${cand}" "agentgraph"; then
    agentgraph_ok=1
    break
  fi
done
if [[ "${agentgraph_ok}" -ne 1 ]]; then
  echo "[trace_report] FATAL: agentgraph not found. Checked:" >&2
  echo "  - \$BAZEL_WORKSPACE_HOME/libs/agent_graph/src (${BAZEL_WS})" >&2
  echo "  - /home/admin/bazel-workspace/libs/agent_graph/src" >&2
  echo "Hint: export BAZEL_WORKSPACE_HOME=/home/admin/bazel-workspace" >&2
  exit 1
fi

for cand in \
  "${BAZEL_WS}/libs/data_models/py" \
  "${REPO_ROOT:+${REPO_ROOT}/libs/data_models/py}" \
  "${HOME}/bazel-workspace/libs/data_models/py" \
  "/home/admin/bazel-workspace/libs/data_models/py"; do
  [[ -z "${cand}" ]] && continue
  if _add_path_if_pkg "${cand}" "data_models"; then
    break
  fi
done

for cand in \
  "${BAZEL_WS}/libs" \
  "${REPO_ROOT:+${REPO_ROOT}/libs}" \
  "${HOME}/bazel-workspace/libs" \
  "/home/admin/bazel-workspace/libs"; do
  [[ -z "${cand}" ]] && continue
  if _add_path_if_pkg "${cand}" "message_queue"; then
    break
  fi
done

for cand in \
  "${BAZEL_WS}/libs/storage/src" \
  "${REPO_ROOT:+${REPO_ROOT}/libs/storage/src}" \
  "${HOME}/bazel-workspace/libs/storage/src" \
  "/home/admin/bazel-workspace/libs/storage/src"; do
  [[ -z "${cand}" ]] && continue
  if _add_path_if_pkg "${cand}" "storage"; then
    break
  fi
done

SCRIPT_PY="${ROOT}/scripts/trace_daily_report_dryrun.py"
if [[ ! -f "${SCRIPT_PY}" ]]; then
  SCRIPT_PY="${SERVICE_ROOT}/scripts/trace_daily_report_dryrun.py"
fi
if [[ ! -f "${SCRIPT_PY}" ]]; then
  echo "[trace_report] FATAL: trace_daily_report_dryrun.py not found" >&2
  exit 1
fi

if [[ "${SERVICE_ROOT}" != "${ROOT}" ]]; then
  export PYTHONPATH="${SERVICE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ -f "${SERVICE_ROOT}/scripts/trace_daily_report_dryrun.py" ]] \
    && [[ ! -d "${ROOT}/app" || ! -f "${ROOT}/app/main.py" ]]; then
    SCRIPT_PY="${SERVICE_ROOT}/scripts/trace_daily_report_dryrun.py"
    cd "${SERVICE_ROOT}"
  fi
fi

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
  elif [[ -n "${REPO_ROOT}" && -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
  elif [[ -x "${SERVICE_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${SERVICE_ROOT}/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

echo "[trace_report] python=${PYTHON}" >&2
echo "[trace_report] SERVICE_ROOT=${SERVICE_ROOT}" >&2
echo "[trace_report] BAZEL_WS=${BAZEL_WS}" >&2

if ! "${PYTHON}" -c "import agentgraph" 2>/dev/null; then
  echo "[trace_report] FATAL: ${PYTHON} cannot import agentgraph" >&2
  echo "[trace_report] PYTHONPATH=${PYTHONPATH}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ $# -eq 0 ]]; then
  [[ -n "${REPORT_NAME:-}" ]] && EXTRA_ARGS+=(--name "${REPORT_NAME}")
  [[ -n "${DATA_DATE:-}" ]] && EXTRA_ARGS+=(--date "${DATA_DATE}")
  [[ -n "${CONFIG_JSON:-}" ]] && EXTRA_ARGS+=(--config-json "${CONFIG_JSON}")
  if [[ "${EXECUTE:-}" == "true" || "${EXECUTE:-}" == "1" ]]; then
    EXTRA_ARGS+=(--execute)
  fi
  if [[ "${FORCE:-}" == "true" || "${FORCE:-}" == "1" ]]; then
    EXTRA_ARGS+=(--force)
  fi
fi

exec "${PYTHON}" "${SCRIPT_PY}" "${EXTRA_ARGS[@]}" "$@"
