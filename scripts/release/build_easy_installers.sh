#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v nix >/dev/null 2>&1; then
  for nix_candidate in \
    /nix/var/nix/profiles/default/bin/nix \
    "$HOME/.nix-profile/bin/nix"; do
    if [ -x "$nix_candidate" ] \
      && [ -x "${nix_candidate%/*}/nix-store" ]; then
      PATH="${nix_candidate%/*}:$PATH"
      export PATH
      break
    fi
  done
fi
command -v nix >/dev/null 2>&1 \
  || { echo "[easy-installer] ERROR: compatible Nix was not found" >&2; exit 1; }

VERSION="${1:-}"
SELECTION="${2:-all}"
if [ -z "$VERSION" ]; then
  VERSION="$(
    nix develop .#complex64-cuda --command python - <<'PY'
from pathlib import Path
import tomllib

print(tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"])
PY
  )"
fi

DIST_DIR="$ROOT_DIR/dist"
SOURCE_PARENT="$DIST_DIR/private-nix-source"
SOURCE_DIR="$SOURCE_PARENT/PyEIDORS-$VERSION"
SOURCE_ZIP="$DIST_DIR/PyEIDORS-$VERSION-pure-nix-source.zip"
SOURCE_SHA="$DIST_DIR/PyEIDORS-$VERSION-pure-nix-source.SHA256SUMS.txt"
TEMPLATE_DIR="$ROOT_DIR/scripts/release/easy-install"
SYSTEM="x86_64-linux"
NIX_FLAGS=(--extra-experimental-features "nix-command flakes")
CLEAN_INTERMEDIATE="${CLEAN_INTERMEDIATE:-1}"
PREPARE_SOURCE="${PREPARE_SOURCE:-1}"

case "$SELECTION" in
  all) editions=(cpu-universal nvidia-sm61 nvidia-modern) ;;
  cpu-universal|nvidia-sm61|nvidia-modern) editions=("$SELECTION") ;;
  *)
    echo "[easy-installer] ERROR: selection must be all, cpu-universal, nvidia-sm61, or nvidia-modern" >&2
    exit 2
    ;;
esac

for template in \
  outer-header.sh.in \
  runtime-common.sh \
  install.sh \
  install-from-local-cache.sh \
  start-pyeidors.sh; do
  [ -f "$TEMPLATE_DIR/$template" ] \
    || { echo "[easy-installer] ERROR: missing template $template" >&2; exit 1; }
done

if [ "$PREPARE_SOURCE" = "1" ] \
  || [ ! -f "$SOURCE_DIR/flake.nix" ] \
  || [ ! -f "$SOURCE_ZIP" ] \
  || [ ! -f "$SOURCE_SHA" ]; then
  echo "[easy-installer] preparing validated source snapshot once"
  RUN_TESTS="${RUN_TESTS:-1}" \
    RUN_WARM="${RUN_WARM:-1}" \
    RUN_CUDA_BUILD="${RUN_CUDA_BUILD:-0}" \
    INCLUDE_CUDA_SM61=1 \
    "$ROOT_DIR/scripts/release/build_private_distribution.sh" "$VERSION"
fi

build_edition() {
  local edition="$1"
  local edition_zh edition_en output_name min_tmp_gib min_home_gib
  local bundle_name bundle_dir flake_ref payload header output output_tmp
  local attr store_path
  local package_attrs=()
  local gpu_modes_json

  case "$edition" in
    cpu-universal)
      edition_zh="CPU 通用版"
      edition_en="CPU Universal"
      output_name="PyEIDORS-$VERSION-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run"
      min_tmp_gib=10
      min_home_gib=30
      package_attrs=(pyeidors pyeidors-complex64 pyeidors-complex)
      gpu_modes_json='[]'
      ;;
    nvidia-sm61)
      edition_zh="NVIDIA SM61 版"
      edition_en="NVIDIA SM61"
      output_name="PyEIDORS-$VERSION-EASY-INSTALL-NVIDIA-SM61-LINUX.run"
      min_tmp_gib=14
      min_home_gib=50
      package_attrs=(
        pyeidors
        pyeidors-complex64
        pyeidors-complex
        pyeidors-cuda-sm61
        pyeidors-complex64-cuda-sm61
      )
      gpu_modes_json='["real", "complex64"]'
      ;;
    nvidia-modern)
      edition_zh="NVIDIA 现代版"
      edition_en="NVIDIA Modern"
      output_name="PyEIDORS-$VERSION-EASY-INSTALL-NVIDIA-MODERN-LINUX.run"
      min_tmp_gib=20
      min_home_gib=70
      package_attrs=(
        pyeidors
        pyeidors-complex64
        pyeidors-complex
        pyeidors-cuda
        pyeidors-cuda-amgx
        pyeidors-complex64-cuda
        pyeidors-complex-cuda
        pyeidors-complex-cuda-amgx
      )
      gpu_modes_json='["real", "complex64", "complex128"]'
      ;;
  esac

  echo "[easy-installer] building $edition_zh"
  NIX_BUILD_CORES=1 \
    NIX_CONFIG="${NIX_CONFIG:-}
cores = 1" \
    BUNDLE_SUFFIX="$edition" \
    PACKAGE_ATTRS_OVERRIDE="${package_attrs[*]}" \
    BUILD_SOURCE_ZIP=0 \
    CREATE_TARBALL=0 \
    MAX_JOBS=1 \
    "$ROOT_DIR/scripts/release/build_binary_cache_bundle.sh" "$VERSION"

  bundle_name="PyEIDORS-$VERSION-fast-install-$edition-$SYSTEM"
  bundle_dir="$DIST_DIR/binary-cache-bundle/$bundle_name"
  [ -d "$bundle_dir/nix-cache" ] \
    || { echo "[easy-installer] ERROR: bundle cache missing" >&2; exit 1; }

  rm -f \
    "$bundle_dir/README_FAST_INSTALL.zh.md" \
    "$bundle_dir/binary-cache-public-key.txt"
  find "$bundle_dir/nix-cache" -type f -name '*.narinfo' \
    -exec sed -i '/^Sig:/d' {} +

  install -m 0755 "$TEMPLATE_DIR/runtime-common.sh" "$bundle_dir/"
  install -m 0755 "$TEMPLATE_DIR/install.sh" "$bundle_dir/"
  install -m 0755 "$TEMPLATE_DIR/install-from-local-cache.sh" "$bundle_dir/"
  install -m 0755 "$TEMPLATE_DIR/start-pyeidors.sh" "$bundle_dir/"

  printf '%s\n' \
    "VERSION='$VERSION'" \
    "EDITION_ID='$edition'" \
    "EDITION_NAME_ZH='$edition_zh'" \
    "EDITION_NAME_EN='$edition_en'" \
    "MIN_HOME_GIB='$min_home_gib'" \
    > "$bundle_dir/edition.conf"

  flake_ref="path:$SOURCE_DIR"
  : > "$bundle_dir/package-map.tsv"
  for attr in "${package_attrs[@]}"; do
    store_path="$(
      nix "${NIX_FLAGS[@]}" path-info \
        --option warn-dirty false "$flake_ref#$attr"
    )"
    [ -x "$store_path/bin/eit-app" ] \
      || { echo "[easy-installer] ERROR: bad package path $attr -> $store_path" >&2; exit 1; }
    printf '%s\t%s\n' "$attr" "$store_path" \
      >> "$bundle_dir/package-map.tsv"
  done

  nix develop .#complex64-cuda --command python - \
    "$ROOT_DIR/docs/EASY_INSTALL_LINUX.zh.md" \
    "$ROOT_DIR/docs/EASY_INSTALL_LINUX.en.md" \
    "$bundle_dir/README_FIRST.zh.md" \
    "$bundle_dir/README_FIRST.en.md" \
    "$VERSION" <<'PY'
from pathlib import Path
import sys

source_zh, source_en, target_zh, target_en, version = sys.argv[1:]
for source, target in ((source_zh, target_zh), (source_en, target_en)):
    text = Path(source).read_text(encoding="utf-8").replace("@VERSION@", version)
    Path(target).write_text(text, encoding="utf-8")
PY

  nix develop .#complex64-cuda --command python - \
    "$bundle_dir/manifest.json" "$bundle_name" "$VERSION" "$edition" \
    "$edition_zh" "$edition_en" "$gpu_modes_json" \
    "${package_attrs[@]}" <<'PY'
from datetime import datetime
import json
from pathlib import Path
import sys

(
    target,
    bundle_name,
    version,
    edition,
    edition_zh,
    edition_en,
    gpu_modes_json,
    *attrs,
) = sys.argv[1:]
data = {
    "name": bundle_name,
    "version": version,
    "system": "x86_64-linux",
    "edition": edition,
    "edition_name_zh": edition_zh,
    "edition_name_en": edition_en,
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "cache_signed": False,
    "cache_key_required": False,
    "nix_conf_edit_required": False,
    "cpu_modes": ["real", "complex64", "complex128"],
    "gpu_modes": json.loads(gpu_modes_json),
    "package_attrs": attrs,
}
Path(target).write_text(
    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

  payload="$DIST_DIR/.$output_name.payload.tar.zst"
  header="$DIST_DIR/.$output_name.header"
  output="$DIST_DIR/$output_name"
  output_tmp="$DIST_DIR/.$output_name.new"
  rm -f "$payload" "$header" "$output_tmp"
  tar -C "$DIST_DIR/binary-cache-bundle" \
    -I 'zstd -T0 -1' -cf "$payload" "$bundle_name"
  payload_sha256="$(sha256sum "$payload")"
  payload_sha256="${payload_sha256%% *}"

  nix develop .#complex64-cuda --command python - \
    "$TEMPLATE_DIR/outer-header.sh.in" \
    "$TEMPLATE_DIR/runtime-common.sh" \
    "$header" \
    "$output_tmp" \
    "$payload" \
    "$bundle_name" \
    "$edition_zh" \
    "$VERSION" \
    "$payload_sha256" \
    "$min_tmp_gib" <<'PY'
from pathlib import Path
import shutil
import sys

(
    template_path,
    common_path,
    header_path,
    output_path,
    payload_path,
    bundle_name,
    edition_zh,
    version,
    payload_sha256,
    min_tmp_gib,
) = sys.argv[1:]
template = Path(template_path).read_text(encoding="utf-8")
common = Path(common_path).read_text(encoding="utf-8")
if common.startswith("#!/usr/bin/env bash\n"):
    common = common.removeprefix("#!/usr/bin/env bash\n")
header = (
    template.replace("# @RUNTIME_COMMON@", common.rstrip())
    .replace("@BUNDLE_NAME@", bundle_name)
    .replace("@EDITION_NAME_ZH@", edition_zh)
    .replace("@VERSION@", version)
    .replace("@PAYLOAD_SHA256@", payload_sha256)
    .replace("@MIN_TMP_GIB@", min_tmp_gib)
)
Path(header_path).write_text(header, encoding="utf-8")
with Path(output_path).open("wb") as destination:
    destination.write(header.encode("utf-8"))
    with Path(payload_path).open("rb") as source:
        shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
PY
  chmod +x "$output_tmp"
  rm -f "$payload" "$header"

  PYEIDORS_EXTRACT_ONLY=1 /bin/bash "$output_tmp"
  mv -f "$output_tmp" "$output"
  echo "[easy-installer] wrote $output"
  sha256sum "$output"

  resolved_bundle="$(realpath -e "$bundle_dir")"
  case "$resolved_bundle" in
    "$DIST_DIR/binary-cache-bundle/"*) rm -rf -- "$resolved_bundle" ;;
    *)
      echo "[easy-installer] ERROR: refusing to clean unexpected path $resolved_bundle" >&2
      exit 1
      ;;
  esac
}

for edition in "${editions[@]}"; do
  build_edition "$edition"
done

nix develop .#complex64-cuda --command python - \
  "$ROOT_DIR/docs/EASY_INSTALL_LINUX.zh.md" \
  "$ROOT_DIR/docs/EASY_INSTALL_LINUX.en.md" \
  "$DIST_DIR/PyEIDORS-$VERSION-EASY-INSTALL-README-ZH.md" \
  "$DIST_DIR/PyEIDORS-$VERSION-EASY-INSTALL-README-EN.md" \
  "$VERSION" <<'PY'
from pathlib import Path
import sys

source_zh, source_en, target_zh, target_en, version = sys.argv[1:]
for source, target in ((source_zh, target_zh), (source_en, target_en)):
    text = Path(source).read_text(encoding="utf-8").replace("@VERSION@", version)
    Path(target).write_text(text, encoding="utf-8")
PY

(
  cd "$DIST_DIR"
  sha256sum PyEIDORS-"$VERSION"-EASY-INSTALL-*-LINUX.run \
    > "PyEIDORS-$VERSION-EASY-INSTALL.SHA256SUMS.txt"
)

nix develop .#complex64-cuda --command python - \
  "$DIST_DIR" "$VERSION" <<'PY'
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

dist = Path(sys.argv[1])
version = sys.argv[2]
specs = (
    ("cpu-universal", f"PyEIDORS-{version}-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run"),
    ("nvidia-modern", f"PyEIDORS-{version}-EASY-INSTALL-NVIDIA-MODERN-LINUX.run"),
    ("nvidia-sm61", f"PyEIDORS-{version}-EASY-INSTALL-NVIDIA-SM61-LINUX.run"),
)
packages = []
for edition, name in specs:
    path = dist / name
    if not path.exists():
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    packages.append(
        {
            "edition": edition,
            "file": name,
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )
manifest = {
    "product": "PyEIDORS",
    "version": version,
    "release_type": "private-pre-publication-preview",
    "system": "x86_64-linux",
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "cache_signed": False,
    "cache_key_required": False,
    "nix_conf_edit_required": False,
    "default_mode": "complex64",
    "packages": packages,
}
(dist / f"PyEIDORS-{version}-EASY-INSTALL-MANIFEST.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

if [ "$CLEAN_INTERMEDIATE" = "1" ]; then
  for target in \
    "$DIST_DIR/binary-cache-bundle" \
    "$SOURCE_PARENT"; do
    if [ -e "$target" ]; then
      resolved_target="$(realpath -e "$target")"
      case "$resolved_target" in
        "$DIST_DIR/"*) rm -rf -- "$resolved_target" ;;
        *)
          echo "[easy-installer] ERROR: refusing to clean unexpected path $resolved_target" >&2
          exit 1
          ;;
      esac
    fi
  done
  rm -f -- "$SOURCE_ZIP" "$SOURCE_SHA"
fi

echo "[easy-installer] final files:"
find "$DIST_DIR" -maxdepth 1 -type f -name "PyEIDORS-$VERSION-EASY-INSTALL*" \
  -printf '%f\t%s bytes\n' | sort
