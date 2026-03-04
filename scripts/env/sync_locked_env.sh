#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

readonly EXPECTED_PY_MM="3.13"
readonly PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
readonly PROFILE_EXTRAS=(torch cuqi dev)

print_profile() {
  cat <<'EOF'
PyEIDORS locked environment profile
- Python interpreter: .venv/bin/python
- Required Python major/minor: 3.13
- uv sync flags: --frozen
- Lock freshness gate: uv lock --check
- Profile extras: torch, cuqi, dev
- Required imports: dolfinx, torch, cuqi, numpy, scipy, pyeidors
EOF
}

ensure_commands() {
  command -v uv >/dev/null 2>&1 || {
    echo "[env-sync] ERROR: uv command not found." >&2
    return 1
  }
  [ -x "$PYTHON_BIN" ] || {
    echo "[env-sync] ERROR: python interpreter not found: $PYTHON_BIN" >&2
    return 1
  }
}

ensure_python_version() {
  local actual
  actual="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "$actual" != "$EXPECTED_PY_MM" ]; then
    echo "[env-sync] ERROR: expected Python $EXPECTED_PY_MM, got $actual ($PYTHON_BIN)." >&2
    echo "[env-sync] Re-enter Nix shell and recreate .venv if needed." >&2
    return 1
  fi
}

build_sync_cmd() {
  SYNC_CMD=(uv sync --python "$PYTHON_BIN" --frozen)
  for extra in "${PROFILE_EXTRAS[@]}"; do
    SYNC_CMD+=(--extra "$extra")
  done
}

run_with_log() {
  local log_file
  log_file="$(mktemp -t pyeidors-env-sync.XXXXXX)"
  if "$@" >"$log_file" 2>&1; then
    rm -f "$log_file"
    return 0
  fi
  cat "$log_file" >&2
  rm -f "$log_file"
  return 1
}

run_import_checks() {
  "$PYTHON_BIN" - <<'PY'
import importlib
import sys
import warnings

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=r"pkg_resources is deprecated as an API",
    module=r"(pkg_resources(\..*)?|setuptools\._vendor\.pkg_resources(\..*)?|cuqi(\..*)?)",
)
warnings.filterwarnings(
    action="ignore",
    category=PendingDeprecationWarning,
    message=r"Importing from numpy\.matlib is deprecated",
    module=r"(numpy\.matlib(\..*)?|cuqi(\..*)?)",
)

required = ["dolfinx", "torch", "cuqi", "numpy", "scipy", "pyeidors"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - runtime check
        missing.append((name, str(exc)))

if missing:
    for name, err in missing:
        print(f"[env-sync] missing import: {name}: {err}", file=sys.stderr)
    raise SystemExit(1)

print("[env-sync] import checks passed")
PY
}

lock_freshness_check() {
  if run_with_log uv lock --check; then
    return 0
  fi
  echo "[env-sync] uv.lock is outdated relative to pyproject metadata." >&2
  echo "[env-sync] Refresh command: uv lock --python \"$PYTHON_BIN\"" >&2
  return 1
}

run_check() {
  ensure_commands
  ensure_python_version
  lock_freshness_check
  build_sync_cmd

  if run_with_log "${SYNC_CMD[@]}" --check; then
    echo "[env-sync] uv environment is synchronized"
  else
    echo "[env-sync] environment drift detected" >&2
    echo "[env-sync] Repair command: scripts/env/sync_locked_env.sh --repair" >&2
    return 1
  fi

  run_import_checks || {
    echo "[env-sync] import validation failed" >&2
    echo "[env-sync] Repair command: scripts/env/sync_locked_env.sh --repair" >&2
    return 1
  }
}

run_repair() {
  ensure_commands
  ensure_python_version
  lock_freshness_check
  build_sync_cmd

  echo "[env-sync] repairing environment with locked profile..."
  "${SYNC_CMD[@]}"
  run_import_checks
  echo "[env-sync] repair completed"
}

mode="${1:---check}"
case "$mode" in
  --print-profile)
    print_profile
    ;;
  --check)
    run_check
    ;;
  --repair)
    run_repair
    ;;
  *)
    echo "Usage: $0 [--check|--repair|--print-profile]" >&2
    exit 2
    ;;
esac
