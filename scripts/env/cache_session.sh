#!/usr/bin/env bash
# Terminal-scoped cache session helper for supported PyEIDORS dev shells.

if [ -n "${_PYEIDORS_CACHE_SESSION_HELPER_LOADED:-}" ]; then
  return 0
fi
_PYEIDORS_CACHE_SESSION_HELPER_LOADED=1

_pyeidors_cache_session_registry_path() {
  local session_dir="${1:-${PYEIDORS_CACHE_SESSION_DIR:-}}"
  printf '%s/.session-dirs\n' "$session_dir"
}

_pyeidors_cache_cleanup_stale_sessions() {
  local requested_root="${1:-}"
  local session_root child base pid
  [ -n "$requested_root" ] || return 0
  session_root="$requested_root/.sessions"
  [ -d "$session_root" ] || return 0

  shopt -s nullglob
  for child in "$session_root"/*; do
    [ -d "$child" ] || continue
    base="$(basename "$child")"
    pid=""
    if [[ "$base" =~ ^session-shellpid([0-9]+)(-|$) ]]; then
      pid="${BASH_REMATCH[1]}"
    elif [[ "$base" =~ ^session-pid([0-9]+)(-|$) ]]; then
      pid="${BASH_REMATCH[1]}"
    fi
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    if [ -n "$pid" ] || [ ! -e "$child/.session-active" ]; then
      rm -rf -- "$child"
    fi
  done
  shopt -u nullglob
  rmdir "$session_root" 2>/dev/null || true
}

_pyeidors_cache_session_cleanup() {
  if [ "${_PYEIDORS_CACHE_SESSION_CLEANED:-0}" = "1" ]; then
    return 0
  fi
  _PYEIDORS_CACHE_SESSION_CLEANED=1

  local registry_path session_dir extra_dir
  session_dir="${PYEIDORS_CACHE_SESSION_DIR:-}"
  registry_path=""
  if [ -n "$session_dir" ]; then
    registry_path="$(_pyeidors_cache_session_registry_path)"
    if [ -f "$registry_path" ]; then
      while IFS= read -r extra_dir || [ -n "$extra_dir" ]; do
        [ -n "$extra_dir" ] || continue
        rm -rf -- "$extra_dir"
        rmdir "$(dirname "$extra_dir")" 2>/dev/null || true
      done < "$registry_path"
    fi
    rm -rf -- "$session_dir"
    rmdir "$(dirname "$session_dir")" 2>/dev/null || true
  fi
  return 0
}

_pyeidors_cache_wrap_deactivate() {
  if ! declare -F deactivate >/dev/null 2>&1; then
    return 0
  fi
  if declare -F __pyeidors_cache_original_deactivate >/dev/null 2>&1; then
    return 0
  fi

  local original_def
  original_def="$(declare -f deactivate)"
  eval "${original_def/deactivate /__pyeidors_cache_original_deactivate }"

  deactivate() {
    local status=0
    if declare -F __pyeidors_cache_original_deactivate >/dev/null 2>&1; then
      __pyeidors_cache_original_deactivate "$@" || status=$?
    fi
    _pyeidors_cache_session_cleanup || true
    return "$status"
  }
}

pyeidors_cache_session_init() {
  local requested_root="${1:-.pyeidors_cache/v2}"
  local owner_pid session_id session_dir

  owner_pid="$$"
  if [ -n "${PYEIDORS_CACHE_SESSION_ID:-}" ] && [ "${PYEIDORS_CACHE_OWNER_PID:-}" = "$owner_pid" ]; then
    return 0
  fi

  mkdir -p -- "$requested_root"
  requested_root="$(cd "$requested_root" && pwd -P)"
  _pyeidors_cache_cleanup_stale_sessions "$requested_root"
  mkdir -p -- "$requested_root/.sessions"

  session_id="session-shellpid${owner_pid}-$(date +%s)-${RANDOM:-0}"
  session_dir="$requested_root/.sessions/$session_id"
  mkdir -p -- "$session_dir"
  : > "$session_dir/.session-active"
  printf '%s\n' "$session_dir" > "$(_pyeidors_cache_session_registry_path "$session_dir")"

  export PYEIDORS_CACHE_SESSION_ID="$session_id"
  export PYEIDORS_CACHE_SESSION_DIR="$session_dir"
  export PYEIDORS_CACHE_REQUESTED_ROOT="$requested_root"
  export PYEIDORS_CACHE_OWNER_PID="$owner_pid"

  _PYEIDORS_CACHE_SESSION_CLEANED=0
  trap '_pyeidors_cache_session_cleanup || true' EXIT HUP INT TERM
  _pyeidors_cache_wrap_deactivate
}
