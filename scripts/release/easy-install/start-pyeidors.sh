#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="$SELF_DIR/edition.conf"
MAP="$SELF_DIR/installed-package-map.tsv"
COMMON="$SELF_DIR/runtime-common.sh"
[ -f "$CONF" ] || { echo "安装配置缺失：$CONF" >&2; exit 1; }
[ -f "$MAP" ] || {
  echo "运行环境映射缺失：$MAP；请重新运行一键安装包。" >&2
  exit 1
}
[ -f "$COMMON" ] || { echo "运行时隔离脚本缺失：$COMMON" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONF"
# shellcheck disable=SC1090
source "$COMMON"

pyeidors_safe_system_path
pyeidors_clean_runtime_environment

MODE="complex64"
FORCE_CPU=0
SHOW_ONLY=0
PASSTHROUGH=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --real) MODE="real"; shift ;;
    --complex64) MODE="complex64"; shift ;;
    --complex128) MODE="complex128"; shift ;;
    --cpu) FORCE_CPU=1; shift ;;
    --show-selection) SHOW_ONLY=1; shift ;;
    --help|-h)
      cat <<EOF
用法：$0 [--real|--complex64|--complex128] [--cpu] [传给 PyEIDORS 的参数]

不带参数时启动 complex64（推荐）。
  --real        实数计算
  --complex64   复数单精度（默认）
  --complex128  复数双精度
  --cpu         强制使用 CPU；GPU 包也可以使用
EOF
      exit 0
      ;;
    --) shift; PASSTHROUGH+=("$@"); break ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

lookup_path() {
  local wanted="$1"
  local attr store_path
  while IFS=$'\t' read -r attr store_path; do
    if [ "$attr" = "$wanted" ]; then
      printf '%s\n' "$store_path"
      return 0
    fi
  done < "$MAP"
  return 1
}

find_nvidia_smi() {
  local candidate
  if [ -n "${PYEIDORS_NVIDIA_SMI:-}" ] \
    && [ -x "$PYEIDORS_NVIDIA_SMI" ]; then
    printf '%s\n' "$PYEIDORS_NVIDIA_SMI"
    return 0
  fi
  for candidate in \
    /usr/bin/nvidia-smi \
    /bin/nvidia-smi \
    /usr/lib/wsl/lib/nvidia-smi \
    /run/current-system/sw/bin/nvidia-smi; do
    if [ -x "$candidate" ] && "$candidate" --help >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

gpu_query_first_line() {
  local query="$1"
  local smi="$2"
  local line=""
  IFS= read -r line < <(
    "$smi" "--query-gpu=$query" --format=csv,noheader,nounits 2>/dev/null
  ) || true
  printf '%s\n' "$line"
}

matching_gpu=0
cap=""
name=""
if smi="$(find_nvidia_smi || true)" && [ -n "$smi" ]; then
  cap="$(gpu_query_first_line compute_cap "$smi")"
  cap="${cap//[[:space:]]/}"
  name="$(gpu_query_first_line name "$smi")"
fi
case "$EDITION_ID" in
  nvidia-sm61)
    if [ "$cap" = "6.1" ] || [[ "$name" =~ GTX[[:space:]]*10[0-9]{2} ]]; then
      matching_gpu=1
    fi
    ;;
  nvidia-modern)
    case "$cap" in
      7.5|8.0|8.6|8.9|9.0|10.0|12.0) matching_gpu=1 ;;
    esac
    if [ "$matching_gpu" -eq 0 ] \
      && [[ "$name" =~ GTX[[:space:]]*16[0-9]{2}|RTX[[:space:]]*(20|30|40|50)[0-9]{2} ]]; then
      matching_gpu=1
    fi
    ;;
esac

case "$MODE" in
  real) cpu_attr="pyeidors" ;;
  complex64) cpu_attr="pyeidors-complex64" ;;
  complex128) cpu_attr="pyeidors-complex" ;;
  *) echo "未知模式：$MODE" >&2; exit 2 ;;
esac

selected_attr="$cpu_attr"
reason="CPU"
if [ "$FORCE_CPU" -eq 0 ] && [ "$matching_gpu" -eq 1 ]; then
  case "$EDITION_ID:$MODE" in
    nvidia-sm61:real)
      selected_attr="pyeidors-cuda-sm61"
      reason="NVIDIA SM61 GPU"
      ;;
    nvidia-sm61:complex64)
      selected_attr="pyeidors-complex64-cuda-sm61"
      reason="NVIDIA SM61 GPU"
      ;;
    nvidia-sm61:complex128)
      reason="CPU（SM61 的 complex128 自动回退）"
      ;;
    nvidia-modern:real)
      selected_attr="pyeidors-cuda-amgx"
      reason="NVIDIA 现代 GPU"
      ;;
    nvidia-modern:complex64)
      selected_attr="pyeidors-complex64-cuda"
      reason="NVIDIA 现代 GPU"
      ;;
    nvidia-modern:complex128)
      selected_attr="pyeidors-complex-cuda"
      reason="NVIDIA 现代 GPU"
      ;;
  esac
fi

store_path="$(lookup_path "$selected_attr" || true)"
[ -n "$store_path" ] || {
  echo "版本 $EDITION_NAME_ZH 不包含 $selected_attr。" >&2
  exit 1
}
[ -x "$store_path/bin/eit-app" ] || {
  echo "运行环境已被清理或损坏：$store_path" >&2
  echo "请重新运行原始 .run 安装包进行修复。" >&2
  exit 1
}

printf 'PyEIDORS %s：%s，模式=%s，运行环境=%s\n' \
  "$VERSION" "$EDITION_NAME_ZH" "$MODE" "$reason"
if [ -n "$name" ]; then
  printf '检测到显卡：%s%s\n' "$name" "${cap:+ (compute capability $cap)}"
elif [[ "$EDITION_ID" == nvidia-* ]]; then
  printf '没有检测到可用 NVIDIA 驱动/显卡，本次安全使用 CPU。\n'
fi
[ "$SHOW_ONLY" -eq 0 ] || exit 0
exec "$store_path/bin/eit-app" "${PASSTHROUGH[@]}"
