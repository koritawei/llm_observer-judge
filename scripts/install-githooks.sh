#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_SRC="$ROOT/.githooks"
HOOKS_DST="$ROOT/.git/hooks"
mkdir -p "$HOOKS_DST"
for hook in prepare-commit-msg commit-msg; do
  install -m 755 "$HOOKS_SRC/$hook" "$HOOKS_DST/$hook"
done
echo "Installed githooks into $HOOKS_DST"
