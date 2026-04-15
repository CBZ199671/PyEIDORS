#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

cd "$REPO_ROOT"

if [[ "$PROFILE" == "gpu" ]]; then
  exec env \
    EIT_APP_GUI_PROFILE="$PROFILE" \
    EIT_APP_SKIP_CUDA_PROBE="$SKIP_CUDA_PROBE" \
    nix develop .#cuda --command bash "$REPO_ROOT/scripts/gui/run_eit_app_inner.sh" "${APP_ARGS[@]}"
fi

exec env \
  EIT_APP_GUI_PROFILE="$PROFILE" \
  EIT_APP_SKIP_CUDA_PROBE="$SKIP_CUDA_PROBE" \
  nix develop --command bash "$REPO_ROOT/scripts/gui/run_eit_app_inner.sh" "${APP_ARGS[@]}"
