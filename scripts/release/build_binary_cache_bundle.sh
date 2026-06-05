#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
  VERSION="$(
    python3 - <<'PY'
from pathlib import Path
import tomllib

project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
print(project["version"])
PY
  )"
fi

SYSTEM="$(nix --extra-experimental-features "nix-command flakes" eval --impure --raw --expr builtins.currentSystem)"
if [ "$SYSTEM" != "x86_64-linux" ]; then
  echo "[binary-cache] ERROR: binary cache bundle is currently supported for x86_64-linux only, got $SYSTEM" >&2
  exit 1
fi

DIST_DIR="$ROOT_DIR/dist"
SOURCE_PARENT="$DIST_DIR/private-nix-source"
SOURCE_DIR="$SOURCE_PARENT/PyEIDORS-$VERSION"
SOURCE_ZIP="$DIST_DIR/PyEIDORS-$VERSION-pure-nix-source.zip"
SOURCE_SHA="$DIST_DIR/PyEIDORS-$VERSION-pure-nix-source.SHA256SUMS.txt"

BUNDLE_NAME="PyEIDORS-$VERSION-fast-install-$SYSTEM"
BUNDLE_PARENT="$DIST_DIR/binary-cache-bundle"
BUNDLE_DIR="$BUNDLE_PARENT/$BUNDLE_NAME"
CACHE_DIR="$BUNDLE_DIR/nix-cache"
TARBALL="$DIST_DIR/$BUNDLE_NAME.tar.zst"
SHA_FILE="$DIST_DIR/$BUNDLE_NAME.SHA256SUMS.txt"

BUILD_SOURCE_ZIP="${BUILD_SOURCE_ZIP:-0}"
BUILD_PACKAGES="${BUILD_PACKAGES:-1}"
MAX_JOBS="${MAX_JOBS:-1}"
CACHE_COMPRESSION="${CACHE_COMPRESSION:-zstd}"
CACHE_COMPRESSION_LEVEL="${CACHE_COMPRESSION_LEVEL:-6}"
ZSTD_LEVEL="${ZSTD_LEVEL:-1}"
CACHE_KEY_NAME="${CACHE_KEY_NAME:-pyeidors-$VERSION-$SYSTEM}"
CACHE_KEY_DIR="${CACHE_KEY_DIR:-$DIST_DIR/binary-cache-keys}"
SECRET_KEY_FILE="${SECRET_KEY_FILE:-$CACHE_KEY_DIR/$CACHE_KEY_NAME.sec}"
PUBLIC_KEY_FILE="${PUBLIC_KEY_FILE:-$CACHE_KEY_DIR/$CACHE_KEY_NAME.pub}"

NIX_FLAGS=(--extra-experimental-features "nix-command flakes")
PACKAGE_ATTRS=(
  pyeidors
  pyeidors-complex
  pyeidors-complex64
  pyeidors-cuda
  pyeidors-complex-cuda
  pyeidors-complex64-cuda
)
APP_ATTRS=(
  eit-app-real-cpu
  eit-app-complex64
  eit-app-complex64-cuda
  eit-app-real-gpu
  eit-app-complex128-cpu
  eit-app-complex128-gpu
)

if [ "$BUILD_SOURCE_ZIP" = "1" ] || [ ! -f "$SOURCE_ZIP" ] || [ ! -f "$SOURCE_SHA" ] || [ ! -f "$SOURCE_DIR/flake.nix" ]; then
  echo "[binary-cache] preparing source zip/staged source tree"
  RUN_TESTS="${RUN_TESTS:-0}" RUN_WARM="${RUN_WARM:-0}" RUN_CUDA_BUILD="${RUN_CUDA_BUILD:-0}" \
    "$ROOT_DIR/scripts/release/build_private_distribution.sh" "$VERSION"
fi

if [ ! -f "$SOURCE_DIR/flake.nix" ]; then
  echo "[binary-cache] ERROR: staged source tree missing: $SOURCE_DIR" >&2
  exit 1
fi

FLAKE_REF="path:$SOURCE_DIR"
PACKAGE_REFS=()
for attr in "${PACKAGE_ATTRS[@]}"; do
  PACKAGE_REFS+=("$FLAKE_REF#$attr")
done

rm -rf "$BUNDLE_DIR"
mkdir -p "$CACHE_DIR"

echo "[binary-cache] source ref: $FLAKE_REF"
echo "[binary-cache] package attrs: ${PACKAGE_ATTRS[*]}"

if [ "$BUILD_PACKAGES" = "1" ]; then
  echo "[binary-cache] ensuring package outputs are built (max-jobs=$MAX_JOBS)"
  nix "${NIX_FLAGS[@]}" build \
    --max-jobs "$MAX_JOBS" \
    --option warn-dirty false \
    --no-link \
    --print-out-paths \
    "${PACKAGE_REFS[@]}" \
    | tee "$BUNDLE_DIR/top-level-store-paths.txt"
else
  echo "[binary-cache] BUILD_PACKAGES=0, collecting package outputs without explicit build"
  nix "${NIX_FLAGS[@]}" path-info \
    --option warn-dirty false \
    "${PACKAGE_REFS[@]}" \
    | tee "$BUNDLE_DIR/top-level-store-paths.txt"
fi

echo "[binary-cache] writing closure manifests"
nix "${NIX_FLAGS[@]}" path-info \
  --recursive \
  --option warn-dirty false \
  "${PACKAGE_REFS[@]}" \
  | sort -u > "$BUNDLE_DIR/closure-store-paths.txt"

nix "${NIX_FLAGS[@]}" path-info \
  --json \
  --json-format 1 \
  --recursive \
  --option warn-dirty false \
  "${PACKAGE_REFS[@]}" \
  > "$BUNDLE_DIR/closure-store-paths.json"

if [ ! -f "$SECRET_KEY_FILE" ] || [ ! -f "$PUBLIC_KEY_FILE" ]; then
  echo "[binary-cache] generating signing key: $CACHE_KEY_NAME"
  mkdir -p "$CACHE_KEY_DIR"
  nix-store --generate-binary-cache-key "$CACHE_KEY_NAME" "$SECRET_KEY_FILE" "$PUBLIC_KEY_FILE"
  chmod 600 "$SECRET_KEY_FILE"
fi

echo "[binary-cache] signing closure paths with $PUBLIC_KEY_FILE"
nix "${NIX_FLAGS[@]}" store sign \
  --key-file "$SECRET_KEY_FILE" \
  --stdin < "$BUNDLE_DIR/closure-store-paths.txt"

echo "[binary-cache] exporting closures to file binary cache"
CACHE_URI="file://$CACHE_DIR?compression=$CACHE_COMPRESSION&compression-level=$CACHE_COMPRESSION_LEVEL"
nix "${NIX_FLAGS[@]}" copy \
  --to "$CACHE_URI" \
  "${PACKAGE_REFS[@]}" \
  --option warn-dirty false

cat > "$BUNDLE_DIR/install-from-local-cache.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$SCRIPT_DIR/nix-cache"
PUBLIC_KEY_FILE="$SCRIPT_DIR/binary-cache-public-key.txt"

if [ ! -f "$CACHE_DIR/nix-cache-info" ]; then
  echo "[pyeidors-cache] ERROR: local Nix cache not found: $CACHE_DIR" >&2
  exit 1
fi
if [ ! -f "$PUBLIC_KEY_FILE" ]; then
  echo "[pyeidors-cache] ERROR: public key file not found: $PUBLIC_KEY_FILE" >&2
  exit 1
fi

PUBLIC_KEY="$(tr -d '\n' < "$PUBLIC_KEY_FILE")"
echo "[pyeidors-cache] importing all store paths from $CACHE_DIR"
if ! nix --extra-experimental-features "nix-command flakes" copy \
  --option extra-trusted-public-keys "$PUBLIC_KEY" \
  --all \
  --from "file://$CACHE_DIR"; then
  cat >&2 <<EOF
[pyeidors-cache] ERROR: Nix refused this local binary cache.

This usually means you are using multi-user Nix and your current user is not
allowed to trust a new binary cache key from the command line.

Ask the machine administrator to add this line to /etc/nix/nix.conf, then
restart nix-daemon:

extra-trusted-public-keys = $PUBLIC_KEY

After that, run this script again:

bash "$SCRIPT_DIR/install-from-local-cache.sh"
EOF
  exit 1
fi
echo "[pyeidors-cache] import complete"
SH
chmod +x "$BUNDLE_DIR/install-from-local-cache.sh"
cp "$PUBLIC_KEY_FILE" "$BUNDLE_DIR/binary-cache-public-key.txt"

cat > "$BUNDLE_DIR/README_FAST_INSTALL.zh.md" <<EOF
# PyEIDORS $VERSION 快速安装二进制缓存包

这个目录包含 PyEIDORS $VERSION 的本地 Nix binary cache。它的作用是把发布者已经编译好的 x86_64-linux 运行环境导入到用户的 /nix/store，避免用户首次运行时长时间编译 FEniCSx、DOLFINx、PETSc、Qt、PyTorch、VTK、CUDA 等大依赖。

## 内容

- nix-cache/：标准 Nix file binary cache。
- install-from-local-cache.sh：把本地 binary cache 导入当前用户的 Nix store。
- binary-cache-public-key.txt：本地 binary cache 的公开签名 key。
- top-level-store-paths.txt：六个 PyEIDORS package 的顶层 store path。
- closure-store-paths.txt：六个 package 共享后的完整闭包路径列表。
- PyEIDORS-$VERSION-pure-nix-source.zip：源码/flake 分发包。
- PyEIDORS-$VERSION-pure-nix-source.SHA256SUMS.txt：源码包校验文件。

## 用户安装步骤

先安装 Nix，并启用 flakes。然后在本目录运行：

\`\`\`bash
bash install-from-local-cache.sh
\`\`\`

如果你的 Nix daemon 不接受命令行传入的 trusted key，请把下面这一行追加到 Nix 配置后重新打开终端，或在 multi-user Nix 中让管理员写入 /etc/nix/nix.conf 并重启 nix-daemon：

\`\`\`bash
extra-trusted-public-keys = $(cat "$PUBLIC_KEY_FILE")
\`\`\`

导入完成后解压源码包：

\`\`\`bash
mkdir -p ~/apps
cd ~/apps
unzip /path/to/$BUNDLE_NAME/PyEIDORS-$VERSION-pure-nix-source.zip
cd PyEIDORS-$VERSION
\`\`\`

CPU 用户启动：

\`\`\`bash
nix run .#eit-app-complex64
\`\`\`

GPU 用户启动：

\`\`\`bash
nix run .#eit-app-complex64-cuda
\`\`\`

如果仍然发生少量构建，通常只是当前源码包路径导致的很小顶层 wrapper 重建；重型依赖应当已经从本地 cache 导入，不应重新编译 CUDA/PETSc/PyTorch/VTK 这类大包。
EOF

cat > "$BUNDLE_DIR/manifest.json" <<EOF
{
  "name": "$BUNDLE_NAME",
  "version": "$VERSION",
  "system": "$SYSTEM",
  "flake_ref": "$FLAKE_REF",
  "package_attrs": [
$(printf '    "%s",\n' "${PACKAGE_ATTRS[@]}" | sed '$ s/,$//')
  ],
  "app_attrs": [
$(printf '    "%s",\n' "${APP_ATTRS[@]}" | sed '$ s/,$//')
  ],
  "created_at": "$(date -Is)"
}
EOF

if [ -f "$SOURCE_ZIP" ]; then
  cp "$SOURCE_ZIP" "$BUNDLE_DIR/"
fi
if [ -f "$SOURCE_SHA" ]; then
  cp "$SOURCE_SHA" "$BUNDLE_DIR/"
fi

echo "[binary-cache] cache size:"
du -sh "$CACHE_DIR" "$BUNDLE_DIR"

echo "[binary-cache] creating $TARBALL"
rm -f "$TARBALL" "$SHA_FILE"
tar -C "$BUNDLE_PARENT" -I "zstd -T0 -$ZSTD_LEVEL" -cf "$TARBALL" "$BUNDLE_NAME"

(
  cd "$DIST_DIR"
  sha256sum "$(basename "$TARBALL")" > "$(basename "$SHA_FILE")"
)

echo "[binary-cache] verifying tarball checksum"
(
  cd "$DIST_DIR"
  sha256sum -c "$(basename "$SHA_FILE")"
)

echo "[binary-cache] wrote $TARBALL"
echo "[binary-cache] wrote $SHA_FILE"
