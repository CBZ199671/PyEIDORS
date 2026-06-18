#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ATTR="${1:-pyeidors-cuda-amgx}"
INTERVAL_SECONDS="${NIX_LIVE_INTERVAL:-5}"
MAX_JOBS="${NIX_LIVE_MAX_JOBS:-1}"
CORES="${NIX_LIVE_CORES:-1}"
LOG_DIR="${NIX_LIVE_LOG_DIR:-$ROOT_DIR/dist/nix-live}"
NO_CLEAR="${NIX_LIVE_NO_CLEAR:-0}"
KEEP_FAILED="${NIX_LIVE_KEEP_FAILED:-0}"

if ! [[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || [ "$INTERVAL_SECONDS" -lt 1 ]; then
  echo "NIX_LIVE_INTERVAL must be a positive integer." >&2
  exit 2
fi

NIX_BIN="${NIX_BIN:-}"
if [ -z "$NIX_BIN" ]; then
  if command -v nix >/dev/null 2>&1; then
    NIX_BIN="$(command -v nix)"
  elif [ -x /nix/var/nix/profiles/default/bin/nix ]; then
    NIX_BIN="/nix/var/nix/profiles/default/bin/nix"
  else
    echo "nix not found in PATH or /nix/var/nix/profiles/default/bin/nix." >&2
    exit 127
  fi
fi

mkdir -p "$LOG_DIR"

safe_attr="$(printf '%s' "$ATTR" | tr -c 'A-Za-z0-9._-' '_')"
started_at="$(date +%Y%m%d-%H%M%S)"
LOG_PATH="$LOG_DIR/$started_at-$safe_attr.log"
STATUS_PATH="$LOG_DIR/$started_at-$safe_attr.status"

NIX_CMD=(
  "$NIX_BIN"
  --extra-experimental-features "nix-command flakes"
  --option warn-dirty false
  --option max-jobs "$MAX_JOBS"
  --option cores "$CORES"
  build ".#$ATTR"
  --no-link
  --print-build-logs
)

if [ "$KEEP_FAILED" = "1" ]; then
  NIX_CMD+=(--keep-failed)
fi

started_epoch="$(date +%s)"

latest_matching_line() {
  local pattern="$1"
  if [ ! -f "$LOG_PATH" ]; then
    return 0
  fi
  if command -v rg >/dev/null 2>&1; then
    rg "$pattern" "$LOG_PATH" | tail -n 1 || true
  else
    grep -E "$pattern" "$LOG_PATH" | tail -n 1 || true
  fi
}

matching_tail() {
  local pattern="$1"
  local lines="$2"
  if [ ! -f "$LOG_PATH" ]; then
    return 0
  fi
  if command -v rg >/dev/null 2>&1; then
    rg "$pattern" "$LOG_PATH" | tail -n "$lines" || true
  else
    grep -E "$pattern" "$LOG_PATH" | tail -n "$lines" || true
  fi
}

render_compile_processes() {
  ps -ww -eo pid,ppid,etime,pcpu,pmem,args --sort=-pcpu \
    2>/dev/null \
    | awk '
      /awk / { next }
      /build_nix_live/ { next }
      /cc1plus|\/g\+\+|\/gcc|\/nvcc|\/ptxas|\/cicc|ninja -j|cmake --build|configure/ {
        line=$0
        kind="process"
        if (line ~ /cc1plus/) kind="cc1plus"
        else if (line ~ /\/nvcc/) kind="nvcc"
        else if (line ~ /\/ptxas/) kind="ptxas"
        else if (line ~ /\/cicc/) kind="cicc"
        else if (line ~ /\/g\+\+/) kind="g++"
        else if (line ~ /\/gcc/) kind="gcc"
        else if (line ~ /ninja -j/) kind="ninja"
        else if (line ~ /cmake --build/) kind="cmake"
        else if (line ~ /configure/) kind="configure"
        printf "pid=%s elapsed=%s cpu=%s%% mem=%s%% kind=%s\n", $1, $3, $4, $5, kind
      }
    ' \
    | sed -n '1,16p'
}

render_dashboard() {
  local state="$1"
  local status="${2:-}"
  local now_epoch elapsed line_count byte_count latest_counter latest_marker warnings errors
  now_epoch="$(date +%s)"
  elapsed="$((now_epoch - started_epoch))"

  if [ "$NO_CLEAR" != "1" ] && [ -t 1 ]; then
    printf '\033[2J\033[H'
  fi

  if [ -f "$LOG_PATH" ]; then
    line_count="$(wc -l < "$LOG_PATH" | tr -d ' ')"
    byte_count="$(stat -c %s "$LOG_PATH")"
    latest_counter="$(latest_matching_line '\[[0-9]+/[0-9]+( [0-9]+%)?\]')"
    latest_marker="$(latest_matching_line '^(building|copying path|error:|warning:|.*> (Running phase|\[[0-9]+/[0-9]+|FAILED|FAILED:|-- Build files|Configuring|Building|Installing))')"
    warnings="$(grep -Ec '(^|[ >])warning:' "$LOG_PATH" || true)"
    errors="$(grep -Ec '(^|[ >])error:' "$LOG_PATH" || true)"
  else
    line_count=0
    byte_count=0
    latest_counter=""
    latest_marker=""
    warnings=0
    errors=0
  fi

  echo "PyEIDORS Nix live build"
  echo "========================"
  printf 'target: .#%s\n' "$ATTR"
  printf 'state: %s%s\n' "$state" "${status:+ status=$status}"
  printf 'elapsed: %02d:%02d:%02d\n' "$((elapsed / 3600))" "$(((elapsed / 60) % 60))" "$((elapsed % 60))"
  printf 'nix: %s\n' "$NIX_BIN"
  printf 'jobs/cores: %s/%s\n' "$MAX_JOBS" "$CORES"
  printf 'keep failed: %s\n' "$KEEP_FAILED"
  printf 'log: %s (%s lines, %s bytes)\n' "$LOG_PATH" "$line_count" "$byte_count"
  printf 'warnings/errors: %s/%s\n' "$warnings" "$errors"

  echo
  echo "Progress"
  echo "--------"
  if [ -n "$latest_counter" ]; then
    printf '%s\n' "$latest_counter"
  else
    echo "No Nix counter emitted yet."
  fi
  if [ -n "$latest_marker" ]; then
    printf '%s\n' "$latest_marker"
  fi

  echo
  echo "Recent Build Markers"
  echo "--------------------"
  matching_tail '^(building|copying path|error:|warning:|.*> (Running phase|\[[0-9]+/[0-9]+|FAILED|FAILED:|-- Build files|Configuring|Building|Installing))' 12

  echo
  echo "Hot Compile Processes"
  echo "---------------------"
  render_compile_processes || true

  echo
  echo "Resources"
  echo "---------"
  free -h | sed -n '1,3p'
  df -h /nix /tmp 2>/dev/null || true

  echo
  echo "Last Log Lines"
  echo "--------------"
  if [ -f "$LOG_PATH" ]; then
    tail -n 12 "$LOG_PATH"
  fi
}

{
  echo "===== Nix live build started: $(date -Is) ====="
  echo "attr=$ATTR"
  echo "command: ${NIX_CMD[*]}"
} | tee -a "$LOG_PATH"

set +e
"${NIX_CMD[@]}" >>"$LOG_PATH" 2>&1 &
build_pid="$!"
set -e

echo "pid=$build_pid" > "$STATUS_PATH"
echo "log=$LOG_PATH" >> "$STATUS_PATH"

trap 'kill "$build_pid" >/dev/null 2>&1 || true' INT TERM

while kill -0 "$build_pid" >/dev/null 2>&1; do
  render_dashboard "running"
  sleep "$INTERVAL_SECONDS"
done

set +e
wait "$build_pid"
status="$?"
set -e

if [ "$status" -eq 0 ]; then
  echo "===== Nix live build finished OK: $(date -Is) =====" >> "$LOG_PATH"
  render_dashboard "ok" "$status"
else
  echo "===== Nix live build failed status=$status: $(date -Is) =====" >> "$LOG_PATH"
  render_dashboard "failed" "$status"
fi

echo "status=$status" >> "$STATUS_PATH"
exit "$status"
