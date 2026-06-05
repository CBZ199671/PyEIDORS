#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${CUDA_SEQUENTIAL_LOG_DIR:-$ROOT_DIR/dist/cuda-sequential}"
SUMMARY_LOG="$LOG_DIR/summary.log"
PROGRESS_LOG="$LOG_DIR/progress.log"
STATUS_FILE="$LOG_DIR/status.tsv"
MAX_JOBS="${CUDA_NIX_MAX_JOBS:-1}"
# CUDA PyTorch source builds can OOM on 32 GiB WSL2 hosts when several nvcc/cicc
# jobs compile flash-attention kernels at once. Keep the default conservative;
# callers can still raise CUDA_NIX_CORES on larger build machines.
CORES="${CUDA_NIX_CORES:-1}"

if [ "$#" -gt 0 ]; then
  ATTRS=("$@")
else
  ATTRS=(pyeidors-cuda pyeidors-complex-cuda pyeidors-complex64-cuda)
fi

NIX_BASE=(
  nix
  --extra-experimental-features "nix-command flakes"
  --option warn-dirty false
  --option max-jobs "$MAX_JOBS"
  --option cores "$CORES"
)

mkdir -p "$LOG_DIR"

log_summary() {
  echo "$*" | tee -a "$SUMMARY_LOG"
}

write_status() {
  local attr="$1"
  local state="$2"
  local status="$3"
  local timestamp
  timestamp="$(date -Is)"
  printf '%s\t%s\t%s\t%s\n' "$timestamp" "$attr" "$state" "$status" >> "$STATUS_FILE"
}

record_progress() {
  local log_path="$1"
  if [ -x "$ROOT_DIR/scripts/release/monitor_cuda_build.sh" ]; then
    TAIL_LINES="${CUDA_MONITOR_TAIL_LINES:-120}" \
      "$ROOT_DIR/scripts/release/monitor_cuda_build.sh" "$log_path" "$PROGRESS_LOG" >/dev/null || true
  fi
}

log_summary "===== CUDA sequential build started: $(date -Is) ====="
log_summary "attrs=${ATTRS[*]}"
log_summary "max-jobs=$MAX_JOBS cores=$CORES log-dir=$LOG_DIR"

for attr in "${ATTRS[@]}"; do
  attr_log="$LOG_DIR/$attr.log"
  log_summary "===== START $attr: $(date -Is) ====="
  write_status "$attr" "start" "0"
  {
    echo "===== START $attr: $(date -Is) ====="
    echo "command: ${NIX_BASE[*]} build .#$attr --no-link --print-build-logs"
  } | tee -a "$attr_log"
  record_progress "$attr_log"

  set +e
  "${NIX_BASE[@]}" build ".#$attr" --no-link --print-build-logs 2>&1 | tee -a "$attr_log"
  status="${PIPESTATUS[0]}"
  set -e

  if [ "$status" -eq 0 ]; then
    log_summary "===== OK $attr: $(date -Is) ====="
    echo "===== OK $attr: $(date -Is) =====" | tee -a "$attr_log"
    write_status "$attr" "ok" "$status"
  else
    log_summary "===== FAIL $attr status=$status: $(date -Is) ====="
    echo "===== FAIL $attr status=$status: $(date -Is) =====" | tee -a "$attr_log"
    write_status "$attr" "fail" "$status"
    record_progress "$attr_log"
    exit "$status"
  fi

  record_progress "$attr_log"
done

log_summary "===== CUDA sequential build finished successfully: $(date -Is) ====="
