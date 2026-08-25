#!/usr/bin/env bash
# Install hermes-compact-context into ~/.hermes/plugins/
# Repo is the source of truth; re-run this after pulling updates, then /reset.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERMES_PLUGINS_DIR:-$HOME/.hermes/plugins}/compact-context"

if [ -e "$DEST" ]; then
  echo "Backing up existing plugin to ${DEST}.bak"
  rm -rf "${DEST}.bak"
  mv "$DEST" "${DEST}.bak"
fi

mkdir -p "$DEST/tests"
cp "$SRC/__init__.py" "$SRC/plugin.yaml" "$DEST/"
cp "$SRC/tests/test_compact_engine.py" "$DEST/tests/"

echo "Installed to $DEST"
echo "Activate:"
echo "  hermes config set context.engine compact-context"
echo "  # plugins.enabled must be a full YAML list — NEVER use '+compact-context'"
echo "  # (that replaces the whole list with a string and silently disables every"
echo "  # plugin). Keep any plugins you already have enabled:"
echo "  hermes config set plugins.enabled '[\"compact-context\"]'"
echo "  # REQUIRED: without in_place, rotation mode reloads the full parent"
echo "  # history in child sessions and re-compresses forever:"
echo "  hermes config set compression.in_place true"
echo "  /reset"
