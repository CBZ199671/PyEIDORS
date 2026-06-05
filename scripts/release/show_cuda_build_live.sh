#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${CUDA_SEQUENTIAL_LOG_DIR:-$ROOT_DIR/dist/cuda-sequential}"
STATUS_FILE="$LOG_DIR/status.tsv"

current_attr=""
if [ -f "$STATUS_FILE" ]; then
  current_attr="$(awk -F '\t' 'END { print $2 }' "$STATUS_FILE")"
fi
if [ -z "$current_attr" ]; then
  current_attr="pyeidors-cuda"
fi

current_log="$LOG_DIR/$current_attr.log"

printf 'time: %s\n' "$(date -Is)"
printf 'session: '
tmux list-sessions 2>/dev/null | rg '^pyeidors-cuda-seq:' || echo 'pyeidors-cuda-seq not running'
printf 'current package: %s\n' "$current_attr"

if [ -f "$STATUS_FILE" ]; then
  echo
  echo 'status:'
  tail -n 6 "$STATUS_FILE"
fi

if [ -f "$current_log" ]; then
  echo
  echo 'log:'
  wc -l "$current_log"
  stat -c 'mtime=%y size=%s bytes' "$current_log"

  last_start_line="$(grep -nF "===== START $current_attr:" "$current_log" | tail -n 1 | cut -d: -f1 || true)"
  log_since_last_start() {
    if [ -n "$last_start_line" ]; then
      tail -n +"$last_start_line" "$current_log"
    else
      cat "$current_log"
    fi
  }

  latest_counter="$(log_since_last_start | grep -Eo '\[[0-9]+/[0-9]+( [0-9]+%)?\]' | tail -n 1 || true)"
  if [ -n "$latest_counter" ]; then
    printf 'latest counter: %s\n' "$latest_counter"
  else
    echo 'latest counter: unavailable from this tool output'
  fi

  latest_build_line="$(log_since_last_start | rg '^(building|copying path|error:|.*> \[[0-9]+/[0-9]+|.*> -- Build files|.*> Running phase|.*> (FAILED|FAILED:))' | tail -n 1 || true)"
  if [ -n "$latest_build_line" ]; then
    printf 'latest log marker: %s\n' "$latest_build_line"
  fi
fi

echo
echo 'active compile targets:'
ps -ww -eo pid,ppid,etime,pcpu,pmem,args --sort=-pcpu \
  | awk '
      /awk / { next }
      /show_cuda_build_live/ { next }
      /cc1plus|\/g\+\+|\/gcc|\/nvcc|\/ptxas|\/cicc|ninja -j|cmake --build/ {
        line=$0
        pid=$1; etime=$3; cpu=$4; mem=$5
        kind="process"
        if (line ~ /cc1plus/) kind="cc1plus"
        else if (line ~ /\/nvcc/) kind="nvcc"
        else if (line ~ /\/ptxas/) kind="ptxas"
        else if (line ~ /\/cicc/) kind="cicc"
        else if (line ~ /\/g\+\+/) kind="g++"
        else if (line ~ /\/gcc/) kind="gcc"
        else if (line ~ /ninja -j/) kind="ninja"
        else if (line ~ /cmake --build/) kind="cmake"

        target=""
        source=""
        if (match(line, / -MT [^ ]+/)) {
          target=substr(line, RSTART + 5, RLENGTH - 5)
        }
        if (match(line, / -o [^ ]+/)) {
          target=substr(line, RSTART + 4, RLENGTH - 4)
        }
        if (match(line, / -c [^ ]+/)) {
          source=substr(line, RSTART + 4, RLENGTH - 4)
        }
        if (match(line, /--orig_src_file_name [^ ]+/)) {
          source=substr(line, RSTART + 21, RLENGTH - 21)
        }
        if (source == "" && match(line, /\/build\/[^ ]+\.(cc|cpp|cu|cxx)/)) {
          source=substr(line, RSTART, RLENGTH)
        }
        if (target == "" && kind == "ninja") {
          target="ninja worker"
        }
        if (target == "" && kind == "cmake") {
          target="cmake build driver"
        }
        printf "pid=%s time=%s cpu=%s%% mem=%s%% kind=%s\n", pid, etime, cpu, mem, kind
        if (target != "") printf "  target: %s\n", target
        if (source != "") printf "  source: %s\n", source
      }
    ' \
  | sed -n '1,80p'

echo
echo 'memory:'
free -h | sed -n '1,3p'
