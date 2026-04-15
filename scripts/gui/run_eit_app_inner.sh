#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROFILE="${EIT_APP_GUI_PROFILE:-cpu}"
SKIP_CUDA_PROBE="${EIT_APP_SKIP_CUDA_PROBE:-0}"
WORKTREE_SRC="${EIT_APP_WORKTREE_SRC:-}"

prepend_pythonpath() {
  local entry="$1"
  case ":${PYTHONPATH:-}:" in
    *":$entry:"*) ;;
    *)
      export PYTHONPATH="$entry${PYTHONPATH:+:$PYTHONPATH}"
      ;;
  esac
}

# Always seed PYTHONPATH with the main repo so that diagnostic scripts (which
# may live only in the main checkout) and fall-back imports keep working.
cd "$REPO_ROOT"
prepend_pythonpath "$REPO_ROOT"
prepend_pythonpath "$REPO_ROOT/src"

# When the outer launcher detected a linked worktree, cd into the worktree
# root and prepend its src/ so that worktree modules win Python import
# resolution.  This lets the nix shell from the main repo execute the
# worktree's work-in-progress code without forcing a re-eval of the flake
# from inside the worktree (which fails silently under nix >= 2.17).
if [[ -n "$WORKTREE_SRC" && -d "$WORKTREE_SRC" ]]; then
  _worktree_root=$(cd "$WORKTREE_SRC/.." && pwd)
  cd "$_worktree_root"
  prepend_pythonpath "$WORKTREE_SRC"
  echo "[run_eit_app_inner] worktree src active: $WORKTREE_SRC" >&2
  echo "[run_eit_app_inner] CWD set to worktree:  $_worktree_root" >&2
fi

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
