#!/usr/bin/env bash
# Enable repo-local Git hooks that strip/block Co-authored-by: Cursor trailers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
chmod +x .githooks/prepare-commit-msg .githooks/commit-msg .githooks/pre-push
git config --local core.hooksPath .githooks
echo "Configured core.hooksPath=.githooks for this repository."
echo "Hooks active: prepare-commit-msg (strip), commit-msg (reject), pre-push (block)."
