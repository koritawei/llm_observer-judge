#!/usr/bin/env bash
# 运维包装：点查 sidecar trace（Lindorm/OSS + tools；可选 --print-notify）
#
# 用法:
#   bash scripts/sidecar.sh --trace-id <id>
#   bash scripts/sidecar.sh --trace-id <id> --project-id <pid>
#   bash scripts/sidecar.sh --trace-id <id> --project-id <pid> --print-notify
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

# 可选：加载 APP_HOME setenv（DB / Redis / Lindorm 等 antx 注入变量）
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

# user-site（base 镜像 pip install --user）
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
  echo "[sidecar] FATAL: agentgraph not found. Checked:" >&2
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
SCRIPT_PY="${ROOT}/scripts/sidecar.py"
if [[ ! -f "${SCRIPT_PY}" ]]; then
  SCRIPT_PY="${SERVICE_ROOT}/scripts/sidecar.py"
fi
if [[ ! -f "${SCRIPT_PY}" ]]; then
  echo "[sidecar] FATAL: sidecar.py not found" >&2
  exit 1
fi

if [[ "${SERVICE_ROOT}" != "${ROOT}" ]]; then
  export PYTHONPATH="${SERVICE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ -f "${SERVICE_ROOT}/scripts/sidecar.py" ]] \
    && [[ ! -d "${ROOT}/app" || ! -f "${ROOT}/app/main.py" ]]; then
    SCRIPT_PY="${SERVICE_ROOT}/scripts/sidecar.py"
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

echo "[sidecar] python=${PYTHON}" >&2
echo "[sidecar] SERVICE_ROOT=${SERVICE_ROOT}" >&2
echo "[sidecar] BAZEL_WS=${BAZEL_WS}" >&2

# 冒烟：确认 agentgraph 可导入
if ! "${PYTHON}" -c "import agentgraph" 2>/dev/null; then
  echo "[sidecar] FATAL: ${PYTHON} cannot import agentgraph" >&2
  echo "[sidecar] PYTHONPATH=${PYTHONPATH}" >&2
  exit 1
fi

exec "${PYTHON}" "${SCRIPT_PY}" "$@"
