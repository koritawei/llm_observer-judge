#!/bin/bash
set -u
workspace="${BAZEL_WORKSPACE_HOME:-}"
storage_src="${workspace}/libs/storage/src"
python_path=""
if [[ -d "${storage_src}/storage" ]]; then
  python_path="${storage_src}"
else
  echo "WARN: libs/storage is optional and was not found" >&2
fi
export PYTHONPATH="${python_path}"
exec python3 -u -m uvicorn benchmark.service.main:create_app
