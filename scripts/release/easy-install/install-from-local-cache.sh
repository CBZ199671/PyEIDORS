#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$SCRIPT_DIR/nix-cache"
PATHS_FILE="$SCRIPT_DIR/top-level-store-paths.txt"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/runtime-common.sh"

say() { printf '[PyEIDORS 缓存] %s\n' "$*"; }
fail() { printf '[PyEIDORS 缓存] 错误：%s\n' "$*" >&2; exit 1; }

[ -f "$CACHE_DIR/nix-cache-info" ] \
  || fail "安装包内的 nix-cache 不完整，请重新下载安装包。"
[ -f "$PATHS_FILE" ] \
  || fail "安装包内缺少 top-level-store-paths.txt，请重新下载安装包。"

NIX_BIN="${PYEIDORS_NIX_BIN:-}"
[ -n "$NIX_BIN" ] \
  || fail "安装主程序没有提供已验证的 Nix 路径，请重新运行外层 .run 文件。"
pyeidors_validate_nix_candidate "$NIX_BIN" \
  || fail "安装主程序提供的 Nix 已失效或版本低于 $PYEIDORS_MIN_NIX_VERSION：$NIX_BIN"
NIX_STORE_BIN="${PYEIDORS_NIX_STORE_BIN:-${NIX_BIN%/*}/nix-store}"
[ "$NIX_STORE_BIN" = "${NIX_BIN%/*}/nix-store" ] && [ -x "$NIX_STORE_BIN" ] \
  || fail "nix 与 nix-store 不是同一套有效安装。"

all_present=1
while IFS= read -r store_path; do
  [ -n "$store_path" ] || continue
  if [ ! -e "$store_path" ]; then
    all_present=0
    break
  fi
done < "$PATHS_FILE"
if [ "$all_present" -eq 1 ]; then
  say "检测到全部运行环境已经存在，跳过重复导入。"
  exit 0
fi

COPY_ARGS=(
  --extra-experimental-features "nix-command flakes"
  copy
  --no-check-sigs
  --all
  --from "file://$CACHE_DIR"
)

say "正在导入安装包内置环境；不需要 cache key，也不会修改 Nix 配置。"
if [ -S /nix/var/nix/daemon-socket/socket ] && [ "$(id -u)" -ne 0 ]; then
  SUDO_BIN="$(pyeidors_resolve_host_tool sudo || true)"
  [ -n "$SUDO_BIN" ] \
    || fail "检测到 multi-user Nix，但系统没有可用 sudo；请让管理员运行安装或导入缓存。"
  say "系统使用 multi-user Nix；接下来只需输入一次 Linux 登录密码。"
  "$SUDO_BIN" -v
  "$SUDO_BIN" -H "$NIX_BIN" "${COPY_ARGS[@]}"
else
  "$NIX_BIN" "${COPY_ARGS[@]}"
fi

while IFS= read -r store_path; do
  [ -n "$store_path" ] || continue
  [ -e "$store_path" ] || fail "导入后仍缺少 $store_path"
done < "$PATHS_FILE"
say "本地环境导入完成。"
