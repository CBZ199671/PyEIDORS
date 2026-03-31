#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

exec nix --option warn-dirty false develop -c \
  uv run python scripts/diagnostics/compare_3d_eidors_alignment.py "$@"
