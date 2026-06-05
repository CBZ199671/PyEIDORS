#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_PATH="${1:-dist/cuda-build.log}"
PROGRESS_PATH="${2:-dist/cuda-build-progress.log}"
INTERVAL_SECONDS="${CUDA_MONITOR_INTERVAL:-60}"
STALE_AFTER_SECONDS="${CUDA_STALE_AFTER_SECONDS:-600}"

mkdir -p "$(dirname "$PROGRESS_PATH")"

last_size=""
last_mtime=""
last_change_epoch="$(date +%s)"

echo "===== CUDA build watcher started: $(date -Is) =====" | tee -a "$PROGRESS_PATH"
echo "log=$LOG_PATH progress=$PROGRESS_PATH interval=${INTERVAL_SECONDS}s stale_after=${STALE_AFTER_SECONDS}s" | tee -a "$PROGRESS_PATH"

has_active_build_process() {
  ps -eo cmd \
    | rg -v "watch_cuda_build|monitor_cuda_build|rg -v|rg -q" \
    | rg -q "nix-daemon [0-9]|/tmp/nix-build-|ninja -j|/nvcc |/ptxas |(^|/)cicc |magma-[0-9]"
}

while true; do
  now_epoch="$(date +%s)"
  stale_seconds=0

  if [ -f "$LOG_PATH" ]; then
    size="$(stat -c %s "$LOG_PATH")"
    mtime="$(stat -c %Y "$LOG_PATH")"
    if [ "$size" != "$last_size" ] || [ "$mtime" != "$last_mtime" ]; then
      last_size="$size"
      last_mtime="$mtime"
      last_change_epoch="$now_epoch"
    fi
    stale_seconds="$((now_epoch - last_change_epoch))"
  fi

  "$ROOT_DIR/scripts/release/monitor_cuda_build.sh" "$LOG_PATH" "$PROGRESS_PATH" >/dev/null || true

  if [ "$stale_seconds" -ge "$STALE_AFTER_SECONDS" ]; then
    {
      echo "===== CUDA build stale warning: $(date -Is) ====="
      echo "log has not changed for ${stale_seconds}s"
      echo
    } | tee -a "$PROGRESS_PATH"
  fi

  if ! has_active_build_process; then
    {
      echo "===== CUDA build watcher stopped: $(date -Is) ====="
      echo "no active CUDA/Nix build process matched"
      echo
    } | tee -a "$PROGRESS_PATH"
    break
  fi

  sleep "$INTERVAL_SECONDS"
done
