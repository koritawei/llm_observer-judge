#!/usr/bin/env bash
# 运维包装：删除指定 run_id 的 runs / run_items 数据（PG 两张表，不重新触发）
#
# 操作：按 run_id 删除 benchmark_run_items 行 + benchmark_runs 行；
# 不清队列 / Redis / inflight 槽位（如需清队列请用 rerun.sh / terminate_run.sh）
#
# 用法（默认 dry-run；加 --execute 才执行）:
#   bash scripts/delete_run_data.sh --run-id ieval_external-20260819
#   bash scripts/delete_run_data.sh --run-id ieval_external-20260819 --execute
#   bash scripts/delete_run_data.sh --run-id ieval_external-20260819 --execute --force
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

# monorepo 本地：ROOT 的上两级是 repo root
REPO_ROOT=""
if [[ -d "${ROOT}/../../libs/agent_graph/src/agentgraph" ]]; then
  REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
fi

# 可选：加载 APP_HOME setenv（DB / Redis 等 antx 注入变量）
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

# ── PYTHONPATH：对齐 bazel_docker_server.sh ─────────────────────────────
export PYTHONPATH="${SERVICE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# user-site
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
  echo "[delete_run_data] FATAL: agentgraph not found. Checked:" >&2
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

# 脚本文件：优先 APP_HOME/scripts，否则 SERVICE_ROOT
SCRIPT_PY="${ROOT}/scripts/delete_run_data.py"
if [[ ! -f "${SCRIPT_PY}" ]]; then
  SCRIPT_PY="${SERVICE_ROOT}/scripts/delete_run_data.py"
fi
if [[ ! -f "${SCRIPT_PY}" ]]; then
  echo "[delete_run_data] FATAL: delete_run_data.py not found" >&2
  exit 1
fi

if [[ "${SERVICE_ROOT}" != "${ROOT}" ]]; then
  export PYTHONPATH="${SERVICE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ -f "${SERVICE_ROOT}/scripts/delete_run_data.py" ]] \
    && [[ ! -d "${ROOT}/app" || ! -f "${ROOT}/app/main.py" ]]; then
    SCRIPT_PY="${SERVICE_ROOT}/scripts/delete_run_data.py"
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

echo "[delete_run_data] python=${PYTHON}" >&2
echo "[delete_run_data] SERVICE_ROOT=${SERVICE_ROOT}" >&2
echo "[delete_run_data] BAZEL_WS=${BAZEL_WS}" >&2

# 冒烟：确认 agentgraph 可导入
if ! "${PYTHON}" -c "import agentgraph" 2>/dev/null; then
  echo "[delete_run_data] FATAL: ${PYTHON} cannot import agentgraph" >&2
  echo "[delete_run_data] PYTHONPATH=${PYTHONPATH}" >&2
  exit 1
fi

exec "${PYTHON}" "${SCRIPT_PY}" "$@"
