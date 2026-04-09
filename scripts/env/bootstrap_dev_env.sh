#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mode="recommended"
action="repair"

usage() {
  cat <<'EOF'
Usage: scripts/env/bootstrap_dev_env.sh [--recommended|--minimal] [--check|--repair]

Modes:
  --recommended  Install required extras plus recommended performance extras.
  --minimal      Install only required extras; performance extras remain optional.

Actions:
  --repair       Synchronize and repair the environment (default).
  --check        Validate the environment without changing installed packages.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --recommended)
      mode="recommended"
      ;;
    --minimal)
      mode="minimal"
      ;;
    --check)
      action="check"
      ;;
    --repair)
      action="repair"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[bootstrap-env] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

export PYEIDORS_ENV_SYNC_INEXACT=1
if [ "$mode" = "recommended" ]; then
  export ENABLE_PERFORMANCE_EXTRAS=1
  echo "[bootstrap-env] Using recommended profile: required extras + performance extras."
else
  export ENABLE_PERFORMANCE_EXTRAS=0
  echo "[bootstrap-env] Using minimal profile: required extras only."
fi

if [ "$action" = "repair" ]; then
  scripts/env/sync_locked_env.sh --repair
else
  scripts/env/sync_locked_env.sh --check
fi

echo "[bootstrap-env] Environment bootstrap completed."
