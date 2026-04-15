#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# -----------------------------------------------------------------------------
# Worktree handling
# -----------------------------------------------------------------------------
# Two issues handled here:
#
# 1. `nix develop` silently exits when invoked from some linked git worktrees
#    (observed with nix >= 2.17 on WSL / Windows).  When we detect a
#    worktree we route the nix shell through the main repository's flake.
#
# 2. Even when launched from the main repo, the form
#    `nix develop --command bash /absolute/path/to/script` has been observed
#    to exit silently in the same environment.  The proven-good form is
#    `nix develop --command bash -c '<inline script>'`.  This launcher uses
#    the inline form, re-execing the committed inner script from inside it.
#
# Together these keep `bash scripts/gui/run_eit_app.sh --cpu` working
# identically from either the main repo or any linked worktree.
NIX_REPO_ROOT="$REPO_ROOT"
EIT_APP_WORKTREE_SRC=""
if [[ -f "$REPO_ROOT/.git" ]]; then
  _common_dir=$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null || true)
  if [[ -n "$_common_dir" ]]; then
    case "$_common_dir" in
      /*) : ;;
      *) _common_dir="$REPO_ROOT/$_common_dir" ;;
    esac
    if [[ -d "$_common_dir" ]]; then
      _common_dir=$(cd "$_common_dir" && pwd)
      _main_candidate=$(cd "$_common_dir/.." && pwd)
      if [[ "$_main_candidate" != "$REPO_ROOT" ]]; then
        NIX_REPO_ROOT="$_main_candidate"
        EIT_APP_WORKTREE_SRC="$REPO_ROOT/src"
        echo "[run_eit_app] worktree detected: $REPO_ROOT" >&2
        echo "[run_eit_app] using nix flake from main repo: $NIX_REPO_ROOT" >&2
      fi
    fi
  fi
fi

PROFILE="cpu"
SKIP_CUDA_PROBE="0"
APP_ARGS=()

while (($# > 0)); do
  case "$1" in
    --cpu|cpu)
      PROFILE="cpu"
      shift
      ;;
    --gpu|gpu)
      PROFILE="gpu"
      shift
      ;;
    --skip-cuda-probe)
      SKIP_CUDA_PROBE="1"
      shift
      ;;
    --)
      shift
      APP_ARGS+=("$@")
      break
      ;;
    *)
      APP_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$NIX_REPO_ROOT"

# The inner script always lives next to this outer script — i.e. in the
# worktree when we were invoked from a worktree, or in the main repo
# otherwise.  Either way, use the absolute path we already computed.
INNER_SCRIPT="$REPO_ROOT/scripts/gui/run_eit_app_inner.sh"

# Compose the inline bash -c payload.  Env vars are exported inside the
# snippet (instead of via `env VAR=val`) because the `env`-prefix form
# interacts badly with nix develop on some hosts.
# `printf '%q'` safely quotes each value against spaces / special chars.
BASH_PAYLOAD=$(cat <<EOF
set -euo pipefail
export EIT_APP_GUI_PROFILE=$(printf '%q' "$PROFILE")
export EIT_APP_SKIP_CUDA_PROBE=$(printf '%q' "$SKIP_CUDA_PROBE")
export EIT_APP_WORKTREE_SRC=$(printf '%q' "$EIT_APP_WORKTREE_SRC")
exec bash $(printf '%q' "$INNER_SCRIPT") "\$@"
EOF
)

if [[ "$PROFILE" == "gpu" ]]; then
  exec nix develop .#cuda --command bash -c "$BASH_PAYLOAD" _ "${APP_ARGS[@]}"
fi

exec nix develop --command bash -c "$BASH_PAYLOAD" _ "${APP_ARGS[@]}"
