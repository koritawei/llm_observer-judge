#!/usr/bin/env bash
# Install GLOBAL Git hooks (~/.githooks) for all repositories on this machine.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOOKS_SRC="$SCRIPT_DIR/../.githooks"
HOOKS_DIR="${HOME}/.githooks"
mkdir -p "$HOOKS_DIR"
for hook in prepare-commit-msg commit-msg pre-push; do
  install -m 755 "$HOOKS_SRC/$hook" "$HOOKS_DIR/$hook"
done
git config --global core.hooksPath "$HOOKS_DIR"
echo "Global core.hooksPath=$HOOKS_DIR"
echo "Hooks installed from $HOOKS_SRC"
