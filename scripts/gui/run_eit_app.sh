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
# Together these keep `bash scripts/gui/run_eit_app.sh --auto` working
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

PROFILE="auto"
PRECISION="${PYEIDORS_GUI_PRECISION:-complex64}"
SKIP_CUDA_PROBE="1"
ENV_SYNC_CACHE="${PYEIDORS_ENV_SYNC_CACHE:-1}"
DRY_RUN="0"
APP_ARGS=()
NIX_OPTS=(--option warn-dirty false)

nix_daemon_proxy_unreachable() {
  local daemon_env
  daemon_env="$(systemctl show nix-daemon.service --property=Environment 2>/dev/null || true)"
  [[ "$daemon_env" == *"127.0.0.1:7897"* ]] || return 1
  if timeout 1 bash -c '</dev/tcp/127.0.0.1/7897' >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

proxy_url_reachable() {
  local url="$1"
  [[ -n "$url" ]] || return 1
  local endpoint="${url#*://}"
  endpoint="${endpoint#*@}"
  endpoint="${endpoint%%/*}"
  local host="${endpoint%%:*}"
  local port="${endpoint##*:}"
  [[ -n "$host" && -n "$port" && "$host" != "$port" ]] || return 1
  timeout 1 bash -c "</dev/tcp/$host/$port" >/dev/null 2>&1
}

reachable_shell_proxy_configured() {
  local candidate
  for candidate in \
    "${https_proxy:-}" \
    "${HTTPS_PROXY:-}" \
    "${http_proxy:-}" \
    "${HTTP_PROXY:-}" \
    "http://127.0.0.1:7890"; do
    if proxy_url_reachable "$candidate"; then
      export http_proxy="${http_proxy:-$candidate}"
      export https_proxy="${https_proxy:-$candidate}"
      export HTTP_PROXY="${HTTP_PROXY:-$candidate}"
      export HTTPS_PROXY="${HTTPS_PROXY:-$candidate}"
      export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
      export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
      return 0
    fi
  done
  return 1
}

prefer_system_env_first() {
  local env_path
  env_path="$(command -v env 2>/dev/null || true)"
  if [[ -n "$env_path" && "$env_path" != "/usr/bin/env" ]]; then
    export PATH="/usr/bin:/bin:$PATH"
    if [[ "${PYEIDORS_LAUNCH_VERBOSE:-0}" == "1" ]]; then
      echo "[run_eit_app] PATH resolves env to $env_path; preferring /usr/bin for this launch" >&2
    fi
  fi
}

prefer_system_env_first

normalize_precision() {
  case "${1,,}" in
    complex128|128|double|float64)
      printf '%s\n' "complex128"
      ;;
    complex64|64|single|float32)
      printf '%s\n' "complex64"
      ;;
    *)
      echo "[run_eit_app] unsupported precision '$1'; expected complex64 or complex128" >&2
      return 2
      ;;
  esac
}

gpu_available() {
  case "${PYEIDORS_GUI_GPU_AVAILABLE:-auto}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    0|false|FALSE|no|NO|off|OFF)
      return 1
      ;;
  esac

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    return 0
  fi
  [[ -e /usr/lib/wsl/lib/libcuda.so || -e /usr/lib/wsl/lib/libcuda.so.1 ]]
}

resolve_launch_profile() {
  local requested="$1"
  local precision="$2"
  NIX_PROFILE="default"
  GUI_PROFILE="cpu"
  case "$requested" in
    auto)
      if gpu_available; then
        GUI_PROFILE="gpu"
        if [[ "$precision" == "complex128" ]]; then
          NIX_PROFILE="complex-cuda"
        else
          NIX_PROFILE="complex64-cuda"
        fi
      else
        GUI_PROFILE="cpu"
        if [[ "$precision" == "complex128" ]]; then
          NIX_PROFILE="complex"
        else
          NIX_PROFILE="complex64"
        fi
      fi
      ;;
    cpu)
      GUI_PROFILE="cpu"
      if [[ "$precision" == "complex128" ]]; then
        NIX_PROFILE="complex"
      else
        NIX_PROFILE="complex64"
      fi
      ;;
    gpu)
      GUI_PROFILE="gpu"
      if [[ "$precision" == "complex128" ]]; then
        NIX_PROFILE="complex-cuda"
      else
        NIX_PROFILE="complex64-cuda"
      fi
      ;;
    real-cpu)
      GUI_PROFILE="cpu"
      NIX_PROFILE="default"
      ;;
    real-gpu)
      GUI_PROFILE="gpu"
      NIX_PROFILE="cuda"
      ;;
    complex-cpu|complex128-cpu)
      GUI_PROFILE="cpu"
      NIX_PROFILE="complex"
      PRECISION="complex128"
      ;;
    complex64-cpu)
      GUI_PROFILE="cpu"
      NIX_PROFILE="complex64"
      PRECISION="complex64"
      ;;
    complex-gpu|complex128-gpu)
      GUI_PROFILE="gpu"
      NIX_PROFILE="complex-cuda"
      PRECISION="complex128"
      ;;
    complex64-gpu)
      GUI_PROFILE="gpu"
      NIX_PROFILE="complex64-cuda"
      PRECISION="complex64"
      ;;
    *)
      echo "[run_eit_app] unsupported profile '$requested'" >&2
      echo "[run_eit_app] use --auto, --cpu, --gpu, --real-cpu, --real-gpu, --complex64-cpu, --complex64-gpu, --complex128-cpu, or --complex128-gpu" >&2
      return 2
      ;;
  esac
}

if nix_daemon_proxy_unreachable; then
  if reachable_shell_proxy_configured; then
    if [[ "${PYEIDORS_LAUNCH_VERBOSE:-0}" == "1" ]]; then
      echo "[run_eit_app] Nix daemon proxy 127.0.0.1:7897 is unreachable, but current shell proxy is reachable; keeping substituters enabled" >&2
    fi
  else
    # The daemon-level proxy is outside this script's environment.  When it
    # points at a closed localhost port and no usable shell proxy exists, Nix
    # emits repeated cache.nixos.org warnings before falling back to local
    # store/build behaviour. Disable lookup for this launch only.
    NIX_OPTS+=(--option substituters "")
    if [[ "${PYEIDORS_LAUNCH_VERBOSE:-0}" == "1" ]]; then
      echo "[run_eit_app] Nix daemon proxy 127.0.0.1:7897 is unreachable; skipping substituter lookup for this launch" >&2
    fi
  fi
fi

while (($# > 0)); do
  case "$1" in
    --auto|auto)
      PROFILE="auto"
      shift
      ;;
    --cpu|cpu)
      PROFILE="cpu"
      shift
      ;;
    --gpu|gpu)
      PROFILE="gpu"
      shift
      ;;
    --real-cpu|real-cpu)
      PROFILE="real-cpu"
      shift
      ;;
    --real-gpu|real-gpu)
      PROFILE="real-gpu"
      shift
      ;;
    --complex-cpu|--complex128-cpu|complex-cpu|complex128-cpu)
      PROFILE="complex128-cpu"
      shift
      ;;
    --complex64-cpu|complex64-cpu)
      PROFILE="complex64-cpu"
      shift
      ;;
    --complex-gpu|--complex128-gpu|complex-gpu|complex128-gpu)
      PROFILE="complex128-gpu"
      shift
      ;;
    --complex64-gpu|complex64-gpu)
      PROFILE="complex64-gpu"
      shift
      ;;
    --precision)
      if (($# < 2)); then
        echo "[run_eit_app] --precision requires complex64 or complex128" >&2
        exit 2
      fi
      PRECISION="$2"
      shift 2
      ;;
    --complex64)
      PRECISION="complex64"
      shift
      ;;
    --complex128)
      PRECISION="complex128"
      shift
      ;;
    --skip-cuda-probe)
      SKIP_CUDA_PROBE="1"
      shift
      ;;
    --probe-cuda|--check-cuda|--verify-cuda)
      SKIP_CUDA_PROBE="0"
      shift
      ;;
    --full-env-check)
      ENV_SYNC_CACHE="0"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
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

PRECISION="$(normalize_precision "$PRECISION")"
resolve_launch_profile "$PROFILE" "$PRECISION"

cd "$NIX_REPO_ROOT"

# The inner script always lives next to this outer script — i.e. in the
# worktree when we were invoked from a worktree, or in the main repo
# otherwise.  Either way, use the absolute path we already computed.
INNER_SCRIPT="$REPO_ROOT/scripts/gui/run_eit_app_inner.sh"

# Compose the inline bash -c payload.  Env vars are exported inside the
# snippet (instead of via `env VAR=val`) because the `env`-prefix form
# interacts badly with nix develop on some hosts.
# `printf '%q'` safely quotes each value against spaces / special chars.
export PYEIDORS_ENV_SYNC_CACHE="$ENV_SYNC_CACHE"
export PYEIDORS_ENV_SYNC_CACHE_TTL_SECONDS="${PYEIDORS_ENV_SYNC_CACHE_TTL_SECONDS:-43200}"
export PYEIDORS_GUI_LAUNCH=1
export PYEIDORS_ENV_SYNC_QUIET_DRIFT="${PYEIDORS_ENV_SYNC_QUIET_DRIFT:-1}"
export PYEIDORS_ENV_SYNC_QUIET_REPAIR="${PYEIDORS_ENV_SYNC_QUIET_REPAIR:-1}"
export UV_NO_PROGRESS="${UV_NO_PROGRESS:-1}"

BASH_PAYLOAD=$(cat <<EOF
set -euo pipefail
export EIT_APP_GUI_PROFILE=$(printf '%q' "$GUI_PROFILE")
export EIT_APP_GUI_REQUESTED_PROFILE=$(printf '%q' "$PROFILE")
export EIT_APP_GUI_RUNTIME_PROFILE=$(printf '%q' "$NIX_PROFILE")
export EIT_APP_GUI_PRECISION=$(printf '%q' "$PRECISION")
export EIT_APP_SKIP_CUDA_PROBE=$(printf '%q' "$SKIP_CUDA_PROBE")
export EIT_APP_WORKTREE_SRC=$(printf '%q' "$EIT_APP_WORKTREE_SRC")
exec bash $(printf '%q' "$INNER_SCRIPT") "\$@"
EOF
)

NIX_TARGET=()
if [[ "$NIX_PROFILE" != "default" ]]; then
  NIX_TARGET=(".#$NIX_PROFILE")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[run_eit_app] requested_profile=$PROFILE"
  echo "[run_eit_app] precision=$PRECISION"
  echo "[run_eit_app] gui_profile=$GUI_PROFILE"
  echo "[run_eit_app] nix_profile=$NIX_PROFILE"
  printf '[run_eit_app] nix command: nix'
  printf ' %q' "${NIX_OPTS[@]}" develop "${NIX_TARGET[@]}" --command bash -c "$BASH_PAYLOAD" _ "${APP_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec nix "${NIX_OPTS[@]}" develop "${NIX_TARGET[@]}" --command bash -c "$BASH_PAYLOAD" _ "${APP_ARGS[@]}"
