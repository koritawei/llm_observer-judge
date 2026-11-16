#!/usr/bin/env bash
# 运维包装：ODPS 抽样导出 search_agent.stream.overview 数据为 CSV 并上传 OSS
#
# ODPS 凭证走 Normandy 资源绑定（normandy-credential-sdk[extension]）：
# AGENT_EVAL_ODPS_AONE_APP_NAME / AGENT_EVAL_ODPS_ENDPOINT 经环境变量下发，
# AK/SK 由 SDK 按资源绑定下发（CT-ODPS-001：禁止硬编码/入仓）；
# OSS 配置（bucket/endpoint/credential）来自 RemoteConfig agent-eval.benchmark.oss。
#
# 用法:
#   bash scripts/export_odps_to_oss.sh                     # T-1 分区，导出+上传
#   bash scripts/export_odps_to_oss.sh --ds 20260907       # 指定分区日补跑
#   bash scripts/export_odps_to_oss.sh --no-upload         # 只导出本地 CSV
#   bash scripts/export_odps_to_oss.sh --dry-run           # 只打印计划，不连 ODPS
#   bash scripts/export_odps_to_oss.sh --oss-ds 20260909   # OSS=T，ODPS 自动 T-1
#   bash scripts/export_odps_to_oss.sh --job search-agent-overview --no-upload
#
# 日期：ODPS 分区 T-1，OSS 路径日期 T（{prefix}/{T}/benchmark_final_results.csv）
# 线上日调度优先用服务内 OdpsExportScheduler（RemoteConfig agent-eval.odps.export +
# Redis 多机锁）；本脚本仅用于补跑 / dry-run / 排障。勿再挂多机 crontab，以免重复写 OSS。
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

# 可选：加载 APP_HOME setenv
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
  echo "[export_odps] FATAL: agentgraph not found. Checked:" >&2
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

# 脚本文件
SCRIPT_PY="${ROOT}/scripts/export_odps_to_oss.py"
if [[ ! -f "${SCRIPT_PY}" ]]; then
  SCRIPT_PY="${SERVICE_ROOT}/scripts/export_odps_to_oss.py"
fi
if [[ ! -f "${SCRIPT_PY}" ]]; then
  echo "[export_odps] FATAL: export_odps_to_oss.py not found" >&2
  exit 1
fi

# 让 python 里的 Path(__file__).parents[1] 指向含 app/ 的目录
if [[ "${SERVICE_ROOT}" != "${ROOT}" ]]; then
  export PYTHONPATH="${SERVICE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ -f "${SERVICE_ROOT}/scripts/export_odps_to_oss.py" ]] \
    && [[ ! -d "${ROOT}/app" || ! -f "${ROOT}/app/main.py" ]]; then
    SCRIPT_PY="${SERVICE_ROOT}/scripts/export_odps_to_oss.py"
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

echo "[export_odps] python=${PYTHON}" >&2
echo "[export_odps] SERVICE_ROOT=${SERVICE_ROOT}" >&2
echo "[export_odps] BAZEL_WS=${BAZEL_WS}" >&2

# 冒烟：确认 agentgraph 可导入
if ! "${PYTHON}" -c "import agentgraph" 2>/dev/null; then
  echo "[export_odps] FATAL: ${PYTHON} cannot import agentgraph" >&2
  echo "[export_odps] PYTHONPATH=${PYTHONPATH}" >&2
  exit 1
fi
# 冒烟：确认 pyodps 可导入（脚本专用可选依赖）
if ! "${PYTHON}" -c "import odps" 2>/dev/null; then
  echo "[export_odps] FATAL: ${PYTHON} cannot import odps (pyodps)" >&2
  echo "[export_odps] Hint: pip install 'pyodps>=0.13.1,<0.14' -i https://artlab.alibaba-inc.com/1/pypi/simple" >&2
  exit 1
fi

exec "${PYTHON}" "${SCRIPT_PY}" "$@"
