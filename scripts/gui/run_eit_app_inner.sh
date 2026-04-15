#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROFILE="${EIT_APP_GUI_PROFILE:-cpu}"
SKIP_CUDA_PROBE="${EIT_APP_SKIP_CUDA_PROBE:-0}"

prepend_pythonpath() {
  local entry="$1"
  case ":${PYTHONPATH:-}:" in
    *":$entry:"*) ;;
    *)
      export PYTHONPATH="$entry${PYTHONPATH:+:$PYTHONPATH}"
      ;;
  esac
}

cd "$REPO_ROOT"
prepend_pythonpath "$REPO_ROOT"
prepend_pythonpath "$REPO_ROOT/src"

python - <<'PY'
import importlib.util
import sys

required = ("pyeidors", "dolfinx", "ufl", "mpi4py", "petsc4py", "PySide6")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "GUI runtime preflight failed. Missing Python modules: "
        + ", ".join(missing)
        + ". Please launch the GUI via `scripts/gui/run_eit_app.sh` from a supported nix shell."
    )
print(
    "GUI runtime preflight OK:",
    {
        "python": sys.executable,
        "path0": sys.path[0],
    },
)
PY

if [[ "$PROFILE" == "gpu" && "$SKIP_CUDA_PROBE" != "1" ]]; then
  python scripts/diagnostics/probe_petsc_cuda.py --require cuda --pretty
fi

exec python -m eit_app.app "$@"
