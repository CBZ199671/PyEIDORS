#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_PATH="${1:-dist/cuda-build.log}"
PROGRESS_PATH="${2:-dist/cuda-build-progress.log}"
TAIL_LINES="${TAIL_LINES:-80}"

mkdir -p "$(dirname "$PROGRESS_PATH")"

timestamp="$(date -Is)"
{
  echo "===== CUDA build monitor: $timestamp ====="

  if [ -f "$LOG_PATH" ]; then
    echo
    echo "== log file =="
    wc -l "$LOG_PATH"
    stat -c 'mtime=%y size=%s' "$LOG_PATH"

    last_start_line="$(grep -nF "===== START " "$LOG_PATH" | tail -n 1 | cut -d: -f1 || true)"
    log_since_last_start() {
      if [ -n "$last_start_line" ]; then
        tail -n +"$last_start_line" "$LOG_PATH"
      else
        cat "$LOG_PATH"
      fi
    }

    echo
    echo "== current build markers =="
    log_since_last_start \
      | grep -E "^(building|copying path|error:|.*> \\[[0-9]+/[0-9]+\\])" \
      | tail -n 40 || true

    echo
    echo "== latest progress counters =="
    log_since_last_start \
      | grep -Eo "\\[[0-9]+/[0-9]+\\]" \
      | tail -n 20 || true

    echo
    echo "== recent log tail =="
    tail -n "$TAIL_LINES" "$LOG_PATH"
  else
    echo "log file not found: $LOG_PATH"
  fi

  echo
  echo "== active build processes =="
  ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd \
    | rg "nix build|nix-daemon|petsc|slepc|dolfinx|vtk|torch|triton|magma|cmake|ninja|gcc|g\\+\\+|nvcc|ptxas|cicc" \
    | sed -n "1,160p" || true

  echo
  echo "== resources =="
  free -h
  df -h / /nix /tmp 2>/dev/null || df -h

  echo
  echo "== nix build roots =="
  ls -ld /tmp/nix-build-* 2>/dev/null | sed -n "1,80p" || true
  echo
} | tee -a "$PROGRESS_PATH"
