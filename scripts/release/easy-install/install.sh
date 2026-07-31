#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/edition.conf"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/runtime-common.sh"

say() { printf '\n[PyEIDORS 一键安装] %s\n' "$*"; }
fail() {
  printf '\n[PyEIDORS 一键安装] 错误：%s\n' "$*" >&2
  printf '[PyEIDORS 一键安装] 安装日志：%s\n' \
    "${LOG_FILE:-尚未建立}" >&2
  exit 1
}

[ "$(uname -s)" = "Linux" ] || fail "只支持 Linux。"
[ "$(uname -m)" = "x86_64" ] \
  || fail "只支持 x86_64 Linux；当前是 $(uname -m)。"
[ "$(id -u)" -ne 0 ] \
  || fail "请使用普通用户运行，不要 sudo bash 安装包。需要权限时脚本会自己询问密码。"

pyeidors_safe_system_path
pyeidors_clean_runtime_environment
MKDIR_BIN="$(pyeidors_resolve_host_tool mkdir)" || fail "系统缺少可用 mkdir。"
DATE_BIN="$(pyeidors_resolve_host_tool date)" || fail "系统缺少可用 date。"
TEE_BIN="$(pyeidors_resolve_host_tool tee)" || fail "系统缺少可用 tee。"
CURL_BIN="$(pyeidors_resolve_host_tool curl)" || fail "系统缺少可用 curl。"
UNZIP_BIN="$(pyeidors_resolve_host_tool unzip)" || fail "系统缺少可用 unzip。"
SHA256_BIN="$(pyeidors_resolve_host_tool sha256sum)" \
  || fail "系统缺少可用 sha256sum。"
DF_BIN="$(pyeidors_resolve_host_tool df)" || fail "系统缺少可用 df。"
AWK_BIN="$(pyeidors_resolve_host_tool awk)" || fail "系统缺少可用 awk。"
MKTEMP_BIN="$(pyeidors_resolve_host_tool mktemp)" \
  || fail "系统缺少可用 mktemp。"
MV_BIN="$(pyeidors_resolve_host_tool mv)" || fail "系统缺少可用 mv。"
RM_BIN="$(pyeidors_resolve_host_tool rm)" || fail "系统缺少可用 rm。"

LOG_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/pyeidors-installer"
"$MKDIR_BIN" -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/PyEIDORS-$VERSION-$EDITION_ID-$("$DATE_BIN" +%Y%m%d-%H%M%S).log"
exec > >("$TEE_BIN" -a "$LOG_FILE") 2>&1

say "版本：PyEIDORS $VERSION $EDITION_NAME_ZH"
say "日志：$LOG_FILE"
say "将忽略用户虚拟环境、Conda、PYTHONPATH 和自定义 CUDA 路径。"

install_nix_if_needed() {
  local nix_bin init_before

  if nix_bin="$(pyeidors_find_nix)"; then
    say "检测到兼容 Nix：$nix_bin（最低要求 $PYEIDORS_MIN_NIX_VERSION）"
    return 0
  fi

  if [ -d /nix ] || [ -e "$HOME/.nix-profile" ]; then
    fail \
      "检测到旧版或损坏的 Nix，但没有找到 Nix >= $PYEIDORS_MIN_NIX_VERSION 与同目录 nix-store。请让管理员升级/修复 Nix；不要同时安装第二套 Nix。"
  fi

  say "没有检测到 Nix；现在使用 Nix 官方安装器安装。此步骤需要联网。"
  if [ -x /usr/bin/systemctl ] \
    && [ "$(/usr/bin/ps -p 1 -o comm= 2>/dev/null | /usr/bin/tr -d '[:space:]')" = "systemd" ]; then
    say "检测到 systemd，采用 multi-user Nix（会询问 Linux 登录密码）。"
    "$CURL_BIN" --fail --location --retry 3 https://nixos.org/nix/install \
      | /bin/sh -s -- --daemon
  else
    say "没有检测到 systemd，采用 single-user Nix。"
    "$CURL_BIN" --fail --location --retry 3 https://nixos.org/nix/install \
      | /bin/sh -s -- --no-daemon
  fi

  if [ -r /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]; then
    init_before="${__ETC_PROFILE_NIX_SOURCED:-}"
    unset __ETC_PROFILE_NIX_SOURCED
    # shellcheck disable=SC1091
    source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh || true
    [ -n "$init_before" ] && export __ETC_PROFILE_NIX_SOURCED="$init_before"
  fi
  if [ -r "$HOME/.nix-profile/etc/profile.d/nix.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/.nix-profile/etc/profile.d/nix.sh" || true
  fi
  pyeidors_safe_system_path
  nix_bin="$(pyeidors_find_nix || true)"
  [ -n "$nix_bin" ] \
    || fail \
      "Nix 安装结束后仍未通过版本/配套检查。请关闭终端、重新打开，再运行同一个 .run 文件。"
}

install_nix_if_needed
NIX_BIN="$(pyeidors_find_nix)"
NIX_STORE_BIN="${NIX_BIN%/*}/nix-store"
export PYEIDORS_NIX_BIN="$NIX_BIN"
export PYEIDORS_NIX_STORE_BIN="$NIX_STORE_BIN"
unset NIX_CONFIG NIX_PATH NIX_REMOTE

say "导入内置二进制环境；不会要求 cache key，也不会修改 nix.conf。"
STORE_PROBE="/"
[ -d /nix ] && STORE_PROBE="/nix"
available_kib="$("$DF_BIN" -Pk "$STORE_PROBE" | "$AWK_BIN" 'NR == 2 { print $4 }')"
required_kib="$((MIN_HOME_GIB * 1024 * 1024))"
if [ "$available_kib" -lt "$required_kib" ]; then
  fail "Nix 存储所在分区空间不足：至少需要 ${MIN_HOME_GIB} GiB 可用空间。"
fi
/bin/bash "$SCRIPT_DIR/install-from-local-cache.sh"

ZIP="$SCRIPT_DIR/PyEIDORS-$VERSION-pure-nix-source.zip"
SHA="$SCRIPT_DIR/PyEIDORS-$VERSION-pure-nix-source.SHA256SUMS.txt"
[ -f "$ZIP" ] || fail "安装包内缺少源码快照。"
[ -f "$SHA" ] || fail "安装包内缺少源码校验文件。"
(cd "$SCRIPT_DIR" && "$SHA256_BIN" -c "$(basename "$SHA")")

INSTALL_PARENT="${PYEIDORS_INSTALL_PARENT:-$HOME/apps}"
INSTALL_ROOT="$INSTALL_PARENT/PyEIDORS-$VERSION"
"$MKDIR_BIN" -p "$INSTALL_PARENT"
STAGE_ROOT="$(
  "$MKTEMP_BIN" -d "$INSTALL_PARENT/.pyeidors-install-$VERSION.XXXXXX"
)"
NEW_ROOT="$STAGE_ROOT/final"
BACKUP=""
SWAPPED=0

cleanup_install() {
  local status="$?"
  trap - EXIT INT TERM
  if [ "$status" -ne 0 ] && [ "$SWAPPED" -eq 1 ]; then
    failed_root="$INSTALL_PARENT/PyEIDORS-$VERSION.failed-$("$DATE_BIN" +%Y%m%d-%H%M%S)"
    [ -e "$INSTALL_ROOT" ] && "$MV_BIN" "$INSTALL_ROOT" "$failed_root"
    if [ -n "$BACKUP" ] && [ -e "$BACKUP" ]; then
      "$MV_BIN" "$BACKUP" "$INSTALL_ROOT"
      printf '[PyEIDORS 一键安装] 已自动恢复安装前版本：%s\n' \
        "$INSTALL_ROOT" >&2
    fi
  fi
  [ ! -e "$STAGE_ROOT" ] || "$RM_BIN" -rf -- "$STAGE_ROOT"
  exit "$status"
}
trap cleanup_install EXIT INT TERM

"$UNZIP_BIN" -q "$ZIP" -d "$STAGE_ROOT/source"
"$MKDIR_BIN" -p "$NEW_ROOT/source" "$NEW_ROOT/.gcroots"
"$MV_BIN" "$STAGE_ROOT/source/PyEIDORS-$VERSION" "$NEW_ROOT/source/"
cp "$SCRIPT_DIR/edition.conf" "$NEW_ROOT/"
cp "$SCRIPT_DIR/runtime-common.sh" "$NEW_ROOT/"
cp "$SCRIPT_DIR/start-pyeidors.sh" "$NEW_ROOT/"
cp "$SCRIPT_DIR/README_FIRST.zh.md" "$NEW_ROOT/README.zh.md"
cp "$SCRIPT_DIR/README_FIRST.en.md" "$NEW_ROOT/README.en.md"

: > "$NEW_ROOT/installed-package-map.tsv"
while IFS=$'\t' read -r attr store_path; do
  [ -n "$attr" ] || continue
  [ -x "$store_path/bin/eit-app" ] \
    || fail "包未正确导入：$attr -> $store_path"
  printf '%s\t%s\n' "$attr" "$store_path" \
    >> "$NEW_ROOT/installed-package-map.tsv"
done < "$SCRIPT_DIR/package-map.tsv"

say "在切换正式目录前检查 real、complex64、complex128 三种 CPU 环境。"
pyeidors_clean_runtime_environment
while IFS=$'\t' read -r attr profile; do
  package_path="$(
    "$AWK_BIN" -F '\t' -v key="$attr" \
      '$1 == key { print $2; exit }' "$NEW_ROOT/installed-package-map.tsv"
  )"
  report="$STAGE_ROOT/doctor-$profile.json"
  CUDA_VISIBLE_DEVICES=-1 "$package_path/bin/eit-backend-doctor" \
    --profile "$profile" --format json > "$report"
  /bin/grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' "$report" \
    || fail "CPU 环境检查失败：$profile。详情见 $report"
  printf '  ✓ %s\n' "$profile"
done <<'EOF'
pyeidors	default
pyeidors-complex64	complex64
pyeidors-complex	complex
EOF

if [ -e "$INSTALL_ROOT" ]; then
  BACKUP="$INSTALL_ROOT.backup-$("$DATE_BIN" +%Y%m%d-%H%M%S)-$$"
  say "发现旧安装；验证完成后安全移动到：$BACKUP"
  "$MV_BIN" "$INSTALL_ROOT" "$BACKUP"
fi
"$MV_BIN" "$NEW_ROOT" "$INSTALL_ROOT"
SWAPPED=1

say "为运行环境建立 Nix GC 保护。"
while IFS=$'\t' read -r attr store_path; do
  [ -n "$attr" ] || continue
  "$NIX_STORE_BIN" --add-root "$INSTALL_ROOT/.gcroots/$attr" \
    -r "$store_path" >/dev/null
done < "$INSTALL_ROOT/installed-package-map.tsv"

say "检查最终硬件选择。"
/bin/bash "$INSTALL_ROOT/start-pyeidors.sh" --show-selection
SWAPPED=0

cat <<EOF

============================================================
安装成功：PyEIDORS $VERSION $EDITION_NAME_ZH
============================================================

以后启动：
  $INSTALL_ROOT/start-pyeidors.sh

默认 complex64；也可使用 --real、--complex128 或 GPU 包的 --cpu。
详细中文说明：
  $INSTALL_ROOT/README.zh.md
英文说明：
  $INSTALL_ROOT/README.en.md
安装日志：
  $LOG_FILE
EOF
